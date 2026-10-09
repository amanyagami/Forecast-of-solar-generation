"""Run forecasters under holdout and rolling-origin protocols and score them."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import pandas as pd

from solarcast import baselines
from solarcast.data import FREQ
from solarcast.features import CLEAR, CLOUDY, Dataset, target
from solarcast.metrics import mae, nrmse, rmse, skill_score
from solarcast.splits import Split, chronological_split, rolling_origin_splits

logger = logging.getLogger(__name__)

HORIZONS = (1, 2, 3, 4)  # 15, 30, 45, 60 minutes
BASELINES = ("persistence", "smart_persistence", "seasonal_naive")
STEP = pd.Timedelta(FREQ)


class LearnedModel(Protocol):
    """Interface implemented by the learned forecasters in :mod:`solarcast.models`."""

    name: str

    def fit_predict(
        self,
        ds: Dataset,
        horizons: Sequence[int],
        fit_end: pd.Timestamp,
        pred_start: pd.Timestamp,
        pred_end: pd.Timestamp,
    ) -> dict[int, pd.Series]:
        """Train on targets before ``fit_end`` and forecast the prediction window."""
        ...


def baseline_forecasts(ds: Dataset, horizon: int) -> dict[str, pd.Series]:
    """Compute all baseline forecasts (indexed by origin) for one horizon.

    Args:
        ds: Dataset.
        horizon: Steps ahead.

    Returns:
        Mapping baseline name -> forecast.
    """
    return {
        "persistence": baselines.persistence(ds.y, horizon),
        "smart_persistence": baselines.smart_persistence(ds.y, ds.env_causal, horizon),
        "seasonal_naive": baselines.seasonal_naive(ds.y, horizon),
    }


def run_block(
    ds: Dataset,
    split: Split,
    horizons: Sequence[int],
    learned: Sequence[LearnedModel],
    gap: int,
    unavailable: set[str] | None = None,
) -> pd.DataFrame:
    """Forecast one test block with every model.

    Args:
        ds: Dataset.
        split: Split whose ``test`` positions index ``ds.frame`` (as target times).
        horizons: Horizons in steps.
        learned: Learned models to train for this block.
        gap: Steps between the end of training targets and the test block.
        unavailable: Names of models that already failed to load; they are skipped and any
            new failure is added to this set so later blocks do not retry.

    Returns:
        Long frame with columns ``origin, target_time, horizon, model, y_true, y_pred``.
    """
    idx = ds.frame.index
    pred_start, pred_end = idx[split.test[0]], idx[split.test[-1]] + STEP
    fit_end = pred_start - STEP * gap
    rows: list[pd.DataFrame] = []
    preds: dict[str, dict[int, pd.Series]] = {m.name: {} for m in learned}
    unavailable = set() if unavailable is None else unavailable
    for m in learned:
        if m.name in unavailable:
            del preds[m.name]
            continue
        try:
            preds[m.name] = m.fit_predict(ds, horizons, fit_end, pred_start, pred_end)
        except (ImportError, OSError) as exc:
            logger.warning("skipping %s: %s", m.name, exc)
            unavailable.add(m.name)
            del preds[m.name]
    for h in horizons:
        truth = target(ds, h)
        tt = idx + STEP * h
        in_block = (tt >= pred_start) & (tt < pred_end)
        per_model = baseline_forecasts(ds, h)
        for name, by_h in preds.items():
            per_model[name] = by_h[h]
        for name, f in per_model.items():
            f = f.reindex(idx)
            keep = in_block & f.notna()
            rows.append(
                pd.DataFrame(
                    {
                        "origin": idx[keep],
                        "target_time": tt[keep],
                        "horizon": h,
                        "model": name,
                        "y_true": truth[keep].to_numpy(),
                        "y_pred": f[keep].to_numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def run_protocol(
    ds: Dataset,
    protocol: str,
    learned: Sequence[LearnedModel],
    horizons: Sequence[int] = HORIZONS,
    n_folds: int = 4,
    test_days: int = 30,
) -> pd.DataFrame:
    """Run a whole evaluation protocol.

    Args:
        ds: Dataset.
        protocol: ``"holdout"`` (one chronological 60/20/20 split, final 20% is the test) or
            ``"rolling"`` (expanding-window rolling origin).
        learned: Learned models.
        horizons: Horizons in steps.
        n_folds: Folds for the rolling protocol.
        test_days: Test block length for the rolling protocol.

    Returns:
        Long prediction frame with an added ``protocol`` column.

    Raises:
        ValueError: For an unknown protocol.
    """
    gap = max(horizons)
    idx = ds.frame.index
    if protocol == "holdout":
        splits = [chronological_split(idx, 0.6, 0.2, gap=gap)]
    elif protocol == "rolling":
        splits = list(rolling_origin_splits(idx, n_folds, test_days, 120, gap=gap))
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    parts = []
    unavailable: set[str] = set()
    for k, sp in enumerate(splits):
        logger.info("%s block %d/%d", protocol, k + 1, len(splits))
        block = run_block(ds, sp, horizons, learned, gap, unavailable)
        block["fold"] = k
        parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    out["protocol"] = protocol
    return out


def score(preds: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Score predictions on daylight steps, split by day type.

    Only daylight target steps where *every* model produced a forecast are scored so all
    models are compared on exactly the same rows. Skill is relative to persistence.

    Args:
        preds: Output of :func:`run_protocol`.
        ds: Dataset (for the daylight mask and day labels).

    Returns:
        Metrics frame with columns ``protocol, model, horizon_min, regime, n, rmse, mae, nrmse,
        skill_vs_persistence``.
    """
    df = preds.copy()
    df["daylight"] = ds.daylight.reindex(df["target_time"]).to_numpy()
    df["regime"] = ds.day_label.reindex(df["target_time"]).to_numpy()
    df = df[df["daylight"] & df["y_true"].notna()]
    n_models = df["model"].nunique()
    key = ["protocol", "fold", "horizon", "target_time"]
    counts = df.groupby(key)["model"].transform("count")
    df = df[counts == n_models]
    records = []
    for regime, sub_all in [("all", df)] + [(r, df[df["regime"] == r]) for r in (CLEAR, CLOUDY)]:
        for (proto, h, model), g in sub_all.groupby(["protocol", "horizon", "model"]):
            ref = sub_all[
                (sub_all["protocol"] == proto)
                & (sub_all["horizon"] == h)
                & (sub_all["model"] == "persistence")
            ]
            r = rmse(g["y_true"], g["y_pred"])
            records.append(
                {
                    "protocol": proto,
                    "model": model,
                    "horizon_min": int(h) * 15,
                    "regime": regime,
                    "n": len(g),
                    "rmse": r,
                    "mae": mae(g["y_true"], g["y_pred"]),
                    "nrmse": nrmse(g["y_true"], g["y_pred"]),
                    "skill_vs_persistence": skill_score(r, rmse(ref["y_true"], ref["y_pred"])),
                }
            )
    return pd.DataFrame.from_records(records)
