"""Learned forecasters: LightGBM (core), a small LSTM (``deep`` extra) and Chronos (optional).

Every model exposes ``fit_predict`` which trains using only targets strictly before
``fit_end`` and returns, per horizon, forecasts indexed by origin time for origins whose target
falls in ``[pred_start, pred_end)``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from solarcast.data import FREQ
from solarcast.features import Dataset, make_features, target

logger = logging.getLogger(__name__)

STEP = pd.Timedelta(FREQ)
GEN_SCALE = 30.0
IRR_SCALE = 1000.0


def _target_times(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Timestamps of the forecast targets for each origin."""
    return index + STEP * horizon


class LightGBMModel:
    """Direct multi-horizon gradient boosting on lag and calendar features."""

    name = "lightgbm"

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.03, seed: int = 0) -> None:
        """Create the model.

        Args:
            n_estimators: Maximum boosting rounds (early stopping may use fewer).
            learning_rate: Shrinkage.
            seed: Random seed.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.seed = seed

    def fit_predict(
        self,
        ds: Dataset,
        horizons: Sequence[int],
        fit_end: pd.Timestamp,
        pred_start: pd.Timestamp,
        pred_end: pd.Timestamp,
    ) -> dict[int, pd.Series]:
        """Train one regressor per horizon and predict the requested window.

        The last 15% of the training days are held out (chronologically) for early stopping.

        Args:
            ds: Dataset.
            horizons: Horizons in steps.
            fit_end: Only rows whose target time is before this are used for fitting.
            pred_start: First target time to predict (inclusive).
            pred_end: Last target time to predict (exclusive).

        Returns:
            Mapping horizon -> forecasts indexed by origin.
        """
        import lightgbm as lgb

        out: dict[int, pd.Series] = {}
        for h in horizons:
            x = make_features(ds, h)
            y = target(ds, h)
            tt = _target_times(x.index, h)
            fit = (tt < fit_end) & y.notna()
            days = x.index[fit].normalize().unique()
            es_start = days[int(len(days) * 0.85)]
            # Early-stopping block ends before fit_end; its targets precede the fit_end boundary
            # and its origins follow the training block, so train/ES do not overlap in target.
            tr = fit & (tt < es_start)
            es = fit & (tt >= es_start)
            reg = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=self.seed,
                verbose=-1,
            )
            reg.fit(
                x[tr],
                y[tr],
                eval_X=x[es],
                eval_y=y[es],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            sel = (tt >= pred_start) & (tt < pred_end)
            pred = pd.Series(reg.predict(x[sel]), index=x.index[sel]).clip(lower=0.0)
            out[h] = pred.rename(self.name)
        return out


class LSTMModel:
    """Small direct multi-horizon LSTM (requires the ``deep`` extra)."""

    name = "lstm"

    def __init__(
        self,
        window: int = 16,
        hidden: int = 32,
        epochs: int = 25,
        batch_size: int = 256,
        lr: float = 2e-3,
        seed: int = 0,
    ) -> None:
        """Create the model.

        Args:
            window: Number of past 15-minute steps fed to the network.
            hidden: LSTM hidden size.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            lr: Adam learning rate.
            seed: Random seed.
        """
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def _inputs(self, ds: Dataset) -> np.ndarray:
        """Per-step input matrix with fixed physical scaling (no fitted statistics)."""
        y = ds.y
        irr = ds.frame["irr"]
        env = ds.env_causal
        idx = ds.frame.index
        tod = (idx.hour * 60 + idx.minute) / (24 * 60)
        feats = np.stack(
            [
                (y / GEN_SCALE).fillna(0.0).to_numpy(),
                y.isna().to_numpy().astype(float),
                (irr / IRR_SCALE).fillna(0.0).to_numpy(),
                irr.isna().to_numpy().astype(float),
                (env / GEN_SCALE).fillna(0.0).to_numpy(),
                np.sin(2 * np.pi * tod),
                np.cos(2 * np.pi * tod),
            ],
            axis=1,
        )
        return feats.astype(np.float32)

    def fit_predict(
        self,
        ds: Dataset,
        horizons: Sequence[int],
        fit_end: pd.Timestamp,
        pred_start: pd.Timestamp,
        pred_end: pd.Timestamp,
    ) -> dict[int, pd.Series]:
        """Train one network predicting all horizons jointly and forecast the window.

        Args:
            ds: Dataset.
            horizons: Horizons in steps.
            fit_end: Only rows whose targets are all before this are used for fitting.
            pred_start: First target time to predict (inclusive).
            pred_end: Last target time to predict (exclusive).

        Returns:
            Mapping horizon -> forecasts indexed by origin.

        Raises:
            ImportError: If torch is not installed.
        """
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        x_all = self._inputs(ds)
        n, w = len(x_all), self.window
        hs = list(horizons)
        ys = np.stack([target(ds, h).to_numpy() / GEN_SCALE for h in hs], axis=1).astype(np.float32)
        idx = ds.frame.index
        origin_ok = np.arange(n) >= w - 1
        last_h = max(hs)
        tt_last = idx + STEP * last_h
        train_rows = np.flatnonzero(origin_ok & (tt_last < fit_end))
        windows = np.lib.stride_tricks.sliding_window_view(x_all, (w, x_all.shape[1]))[:, 0]
        # windows[i] covers rows i..i+w-1 -> origin index i + w - 1
        xw = torch.from_numpy(windows[train_rows - (w - 1)].copy())
        yt = torch.from_numpy(ys[train_rows])

        class Net(nn.Module):
            def __init__(self, n_in: int, hidden: int, n_out: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
                self.head = nn.Linear(hidden, n_out)

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                seq, _ = self.lstm(inp)
                return self.head(seq[:, -1])

        net = Net(x_all.shape[1], self.hidden, len(hs))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(xw), generator=gen)
            for i in range(0, len(perm), self.batch_size):
                b = perm[i : i + self.batch_size]
                pred = net(xw[b])
                mask = ~torch.isnan(yt[b])
                loss = ((pred - torch.nan_to_num(yt[b])) ** 2 * mask).sum() / mask.sum().clamp(
                    min=1
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        rows = np.flatnonzero(origin_ok)
        with torch.no_grad():
            preds = net(torch.from_numpy(windows[rows - (w - 1)].copy())).numpy() * GEN_SCALE
        out: dict[int, pd.Series] = {}
        for j, h in enumerate(hs):
            tt = _target_times(idx[rows], h)
            sel = (tt >= pred_start) & (tt < pred_end)
            out[h] = pd.Series(preds[sel, j], index=idx[rows][sel], name=self.name).clip(lower=0.0)
        return out


class ChronosModel:
    """Zero-shot Chronos foundation model (requires the ``foundation`` extra and weights)."""

    name = "chronos"

    def __init__(
        self, model_id: str = "amazon/chronos-t5-tiny", context: int = 192, num_samples: int = 10
    ) -> None:
        """Create the wrapper (weights are loaded lazily in ``fit_predict``).

        Args:
            model_id: Hugging Face model id.
            context: Number of past steps given as context.
            num_samples: Samples per forecast; the median is used.
        """
        self.model_id = model_id
        self.context = context
        self.num_samples = num_samples

    def fit_predict(
        self,
        ds: Dataset,
        horizons: Sequence[int],
        fit_end: pd.Timestamp,
        pred_start: pd.Timestamp,
        pred_end: pd.Timestamp,
    ) -> dict[int, pd.Series]:
        """Forecast zero-shot; no training takes place.

        Args:
            ds: Dataset.
            horizons: Horizons in steps.
            fit_end: Unused (zero-shot) but kept for interface symmetry.
            pred_start: First target time to predict (inclusive).
            pred_end: Last target time to predict (exclusive).

        Returns:
            Mapping horizon -> forecasts indexed by origin.

        Raises:
            ImportError: If chronos or torch are missing.
            OSError: If the weights cannot be downloaded.
        """
        del fit_end
        import torch
        from chronos import ChronosPipeline

        pipe = ChronosPipeline.from_pretrained(
            self.model_id, device_map="cpu", torch_dtype=torch.float32
        )
        y = ds.y.ffill().fillna(0.0).to_numpy(dtype=np.float32)
        idx = ds.frame.index
        hmax = max(horizons)
        out: dict[int, list[float]] = {h: [] for h in horizons}
        origins: dict[int, list[pd.Timestamp]] = {h: [] for h in horizons}
        cand = np.flatnonzero(
            (idx + STEP * min(horizons) >= pred_start) & (idx >= idx[self.context])
        )
        cand = cand[idx[cand] + STEP * min(horizons) < pred_end]
        for i in range(0, len(cand), 64):
            pos = cand[i : i + 64]
            ctx = [torch.from_numpy(y[p - self.context + 1 : p + 1]) for p in pos]
            samples = pipe.predict(ctx, hmax, num_samples=self.num_samples)
            med = np.median(samples.numpy(), axis=1)
            for h in horizons:
                for k, p in enumerate(pos):
                    tt = idx[p] + STEP * h
                    if pred_start <= tt < pred_end:
                        out[h].append(float(med[k, h - 1]))
                        origins[h].append(idx[p])
        return {
            h: pd.Series(out[h], index=pd.DatetimeIndex(origins[h]), name=self.name).clip(lower=0.0)
            for h in horizons
        }
