"""Data-driven clear-sky envelope, clearness ratio and clear/cloudy day labels.

Design choice: pvlib was *not* used. A physical clear-sky model needs site calibration of the
PV system (capacity, tilt, losses) to be comparable with metered kWh, which is unknown here.
Instead the clear-sky envelope is learned from the data: for every time-of-day slot, the upper
quantile (default 0.95) of that slot's values over a rolling window of days. This automatically
absorbs panel size, soiling and the seasonal sun path, at the price of being biased low in
persistently overcast periods (the Indian south-west monsoon).

Two flavours exist:

* ``centered`` window (includes the day itself and the future): used only for *labelling*
  days after the fact, which is an offline analysis step.
* ``trailing`` window (strictly previous days): causal, safe to use as a forecast feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarcast.data import STEPS_PER_DAY

CLEAR = "clear"
CLOUDY = "cloudy"
UNKNOWN = "unknown"


def _to_day_slot_matrix(series: pd.Series) -> pd.DataFrame:
    """Reshape a 15-minute series into a (day x slot) matrix.

    Args:
        series: Series on a regular, gap-free 15-minute index.

    Returns:
        Frame indexed by normalised day with ``STEPS_PER_DAY`` columns (NaN padded).
    """
    day = series.index.normalize()
    slot = (series.index.hour * 60 + series.index.minute) // 15
    return series.groupby([day, slot]).first().unstack().reindex(columns=range(STEPS_PER_DAY))


def clear_sky_envelope(
    series: pd.Series,
    window_days: int = 31,
    quantile: float = 0.95,
    mode: str = "centered",
    min_days: int = 5,
) -> pd.Series:
    """Estimate a clear-sky envelope as a per-slot rolling upper quantile.

    Args:
        series: Generation (or irradiation) on a regular 15-minute grid.
        window_days: Rolling window length in days.
        quantile: Upper quantile defining the envelope.
        mode: ``"centered"`` (offline labelling) or ``"trailing"`` (strictly previous days,
            causal).
        min_days: Minimum number of valid days required in a window; otherwise the
            expanding quantile of the available history is used, or NaN when none exists.

    Returns:
        Envelope aligned with ``series.index``.

    Raises:
        ValueError: If ``mode`` is not recognised.
    """
    if mode not in {"centered", "trailing"}:
        raise ValueError(f"mode must be 'centered' or 'trailing', got {mode!r}")
    mat = _to_day_slot_matrix(series)
    mat = mat.reindex(pd.date_range(mat.index.min(), mat.index.max(), freq="D"))
    if mode == "centered":
        roll = mat.rolling(window_days, center=True, min_periods=min_days)
        env = roll.quantile(quantile)
    else:
        past = mat.shift(1)
        roll = past.rolling(window_days, min_periods=min_days)
        env = roll.quantile(quantile)
        # Warm-up: until ``min_days`` days exist there is no causal estimate -> NaN.
    day = series.index.normalize()
    slot = (series.index.hour * 60 + series.index.minute) // 15
    values = env.to_numpy()[env.index.get_indexer(day), np.asarray(slot)]
    return pd.Series(values, index=series.index, name="envelope")


def clearness_ratio(
    series: pd.Series, envelope: pd.Series, min_envelope_frac: float = 0.05
) -> pd.Series:
    """Compute the clearness ratio ``series / envelope`` in daylight.

    Args:
        series: Observed generation.
        envelope: Clear-sky envelope aligned with ``series``.
        min_envelope_frac: Fraction of the envelope maximum below which the sun is considered
            too low for a stable ratio (result is NaN there).

    Returns:
        Ratio series, clipped to ``[0, 1.5]``.
    """
    floor = min_envelope_frac * float(np.nanmax(envelope.to_numpy()))
    ratio = series / envelope.where(envelope > floor)
    return ratio.clip(0.0, 1.5)


def daylight_mask(envelope: pd.Series, min_envelope_frac: float = 0.05) -> pd.Series:
    """Boolean mask of daylight time steps.

    Args:
        envelope: Clear-sky envelope.
        min_envelope_frac: Fraction of the envelope maximum defining daylight.

    Returns:
        Boolean series, True in daylight.
    """
    return envelope > min_envelope_frac * float(np.nanmax(envelope.to_numpy()))


def label_days(
    series: pd.Series,
    envelope: pd.Series,
    clear_threshold: float = 0.8,
    min_coverage: float = 0.9,
) -> pd.Series:
    """Label each calendar day as clear, cloudy or unknown.

    The daily clearness index is ``sum(series) / sum(envelope)`` over daylight steps. A day is
    ``clear`` when the index is at least ``clear_threshold`` and ``cloudy`` otherwise. Days where
    less than ``min_coverage`` of the daylight steps are observed are ``unknown``.

    Args:
        series: Observed generation (NaN where missing).
        envelope: Clear-sky envelope aligned with ``series`` (use the centered mode).
        clear_threshold: Daily clearness index at or above which a day is clear.
        min_coverage: Minimum observed fraction of daylight steps.

    Returns:
        Series indexed by normalised day with values ``clear``, ``cloudy`` or ``unknown``.
    """
    day_mask = daylight_mask(envelope)
    obs = series.where(day_mask)
    env = envelope.where(day_mask)
    day = series.index.normalize()
    coverage = obs.notna().groupby(day).sum() / day_mask.groupby(day).sum().clip(lower=1)
    # Only sum steps where both are valid so partial days are not biased.
    both = obs.notna() & env.notna()
    num = obs.where(both).groupby(day).sum()
    den = env.where(both).groupby(day).sum()
    index = num / den.where(den > 0)
    labels = pd.Series(CLOUDY, index=index.index, dtype=object)
    labels[index >= clear_threshold] = CLEAR
    labels[(coverage < min_coverage) | index.isna()] = UNKNOWN
    labels.name = "day_label"
    return labels


def label_per_step(labels: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Broadcast day labels to every time step.

    Args:
        labels: Output of :func:`label_days`.
        index: Time index to broadcast to.

    Returns:
        Label for the day containing each timestamp.
    """
    mapped = labels.reindex(index.normalize())
    return pd.Series(mapped.to_numpy(), index=index, name="day_label")
