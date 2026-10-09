"""Leakage-free feature construction for direct multi-horizon forecasting.

A forecast *origin* ``t`` produces features from data observed at or before ``t`` plus
deterministic calendar terms and the causal (trailing) clear-sky envelope. The target is the
generation at ``t + horizon`` steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from solarcast.data import ENV_FLOOR_KWH, FREQ, GEN_COL, IRR_COL, STEPS_PER_DAY
from solarcast.labeling import (
    CLEAR,
    CLOUDY,
    UNKNOWN,
    clear_sky_envelope,
    daylight_mask,
    label_days,
    label_per_step,
)

GEN_LAGS = (0, 1, 2, 3, 4, 6, 8, 12)
IRR_LAGS = (0, 1, 2, 3)

__all__ = ["CLEAR", "CLOUDY", "UNKNOWN", "Dataset", "build_dataset", "make_features", "target"]


@dataclass
class Dataset:
    """Container with everything the forecasters and the evaluation need.

    Attributes:
        frame: Aligned ``gen_kwh``/``irr`` frame on a regular grid.
        env_causal: Trailing clear-sky envelope (safe as a feature).
        env_label: Centered envelope, used only to label days and mask night.
        day_label: Per-step label (clear/cloudy/unknown) of the containing day.
        daylight: Per-step daylight mask.
    """

    frame: pd.DataFrame
    env_causal: pd.Series
    env_label: pd.Series
    day_label: pd.Series
    daylight: pd.Series

    @property
    def y(self) -> pd.Series:
        """Observed generation."""
        return self.frame[GEN_COL]


def build_dataset(
    frame: pd.DataFrame, window_days: int = 31, quantile: float = 0.95, clear_threshold: float = 0.8
) -> Dataset:
    """Derive envelopes, day labels and the daylight mask for a frame.

    Args:
        frame: Aligned frame from :func:`solarcast.data.load_dataset`.
        window_days: Envelope window in days.
        quantile: Envelope quantile.
        clear_threshold: Daily clearness index separating clear from cloudy days.

    Returns:
        A :class:`Dataset`.
    """
    y = frame[GEN_COL]
    env_label = clear_sky_envelope(y, window_days, quantile, mode="centered")
    env_causal = clear_sky_envelope(y, window_days, quantile, mode="trailing")
    labels = label_days(y, env_label, clear_threshold)
    return Dataset(
        frame=frame,
        env_causal=env_causal,
        env_label=env_label,
        day_label=label_per_step(labels, frame.index),
        daylight=daylight_mask(env_label),
    )


def target(ds: Dataset, horizon: int) -> pd.Series:
    """Generation ``horizon`` steps after each origin (indexed by origin).

    Args:
        ds: Dataset.
        horizon: Steps ahead.

    Returns:
        Target series aligned with origins.
    """
    return ds.y.shift(-horizon).rename("target")


def make_features(ds: Dataset, horizon: int) -> pd.DataFrame:
    """Build the feature matrix for one horizon, indexed by origin time.

    Args:
        ds: Dataset.
        horizon: Steps ahead (used for the target-time calendar and envelope terms).

    Returns:
        Feature frame; NaN is left in place for LightGBM to handle.
    """
    y, irr, env = ds.y, ds.frame[IRR_COL], ds.env_causal
    cols: dict[str, pd.Series] = {}
    for k in GEN_LAGS:
        cols[f"y_lag{k}"] = y.shift(k)
    for k in IRR_LAGS:
        cols[f"irr_lag{k}"] = irr.shift(k)
    cols["y_same_time_yesterday"] = y.shift(STEPS_PER_DAY - horizon)
    kc = (y / env.where(env > ENV_FLOOR_KWH)).clip(0.0, 1.5)
    cols["kc"] = kc
    cols["kc_mean4"] = kc.rolling(4, min_periods=2).mean()
    cols["kc_std4"] = kc.rolling(4, min_periods=2).std()
    cols["y_mean4"] = y.rolling(4, min_periods=2).mean()
    cols["env_now"] = env
    cols["env_target"] = env.shift(-horizon)
    t_idx = ds.frame.index + pd.Timedelta(FREQ) * horizon
    tod = (t_idx.hour * 60 + t_idx.minute) / (24 * 60)
    doy = t_idx.dayofyear / 366.0
    cols["tod_sin"] = pd.Series(np.sin(2 * np.pi * tod), index=ds.frame.index)
    cols["tod_cos"] = pd.Series(np.cos(2 * np.pi * tod), index=ds.frame.index)
    cols["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy), index=ds.frame.index)
    cols["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy), index=ds.frame.index)
    return pd.DataFrame(cols, index=ds.frame.index)
