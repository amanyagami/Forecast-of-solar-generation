"""Reference forecasters. All return forecasts indexed by *origin* time.

Row ``t`` of the output is the forecast, issued at ``t`` using only data up to ``t``, for the
value at ``t + horizon`` steps.
"""

from __future__ import annotations

import pandas as pd

from solarcast.data import ENV_FLOOR_KWH, STEPS_PER_DAY


def persistence(y: pd.Series, horizon: int) -> pd.Series:
    """Predict that the future equals the present.

    Args:
        y: Observed series on a regular grid.
        horizon: Forecast horizon in steps (unused apart from validation).

    Returns:
        Forecast indexed by origin time.

    Raises:
        ValueError: If ``horizon`` is not positive.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return y.rename("persistence")


def smart_persistence(
    y: pd.Series, envelope: pd.Series, horizon: int, fallback_ratio: float = 1.0
) -> pd.Series:
    """Persist the clearness ratio rather than the raw value.

    ``y_hat(t+h) = kc(t) * envelope(t+h)`` with ``kc(t) = y(t) / envelope(t)``. When the sun is
    too low for a stable ratio, ``fallback_ratio`` (default clear sky) is used. The envelope must
    be causal (trailing mode) to keep the forecast honest.

    Args:
        y: Observed series.
        envelope: Causal clear-sky envelope on the same index.
        horizon: Forecast horizon in steps.
        fallback_ratio: Ratio used when it is undefined at the origin.

    Returns:
        Forecast indexed by origin time (NaN where the envelope is unknown).

    Raises:
        ValueError: If ``horizon`` is not positive.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    dark = ~(envelope > ENV_FLOOR_KWH)
    kc = (y / envelope.where(~dark)).clip(0.0, 1.5)
    kc = kc.mask(dark, fallback_ratio)
    return (kc * envelope.shift(-horizon)).rename("smart_persistence")


def seasonal_naive(y: pd.Series, horizon: int, period: int = STEPS_PER_DAY) -> pd.Series:
    """Predict the value observed one period (default one day) before the target time.

    Args:
        y: Observed series.
        horizon: Forecast horizon in steps.
        period: Seasonal period in steps; must exceed ``horizon``.

    Returns:
        Forecast indexed by origin time.

    Raises:
        ValueError: If ``period`` does not exceed ``horizon``.
    """
    if period <= horizon:
        raise ValueError("period must exceed horizon so the forecast uses only past data")
    return y.shift(period - horizon).rename("seasonal_naive")
