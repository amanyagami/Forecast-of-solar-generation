"""Point-forecast error metrics and skill scores."""

from __future__ import annotations

import numpy as np


def _clean(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert to float arrays and drop pairs containing NaN."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    keep = ~(np.isnan(t) | np.isnan(p))
    return t[keep], p[keep]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (NaN pairs ignored; NaN if nothing is left).

    Args:
        y_true: Observed values.
        y_pred: Forecasts.

    Returns:
        The RMSE.
    """
    t, p = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if t.size else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error (NaN pairs ignored).

    Args:
        y_true: Observed values.
        y_pred: Forecasts.

    Returns:
        The MAE.
    """
    t, p = _clean(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if t.size else float("nan")


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalised by the mean of the observations.

    Args:
        y_true: Observed values.
        y_pred: Forecasts.

    Returns:
        ``RMSE / mean(y_true)``; NaN when the mean is not positive.
    """
    t, p = _clean(y_true, y_pred)
    if not t.size or t.mean() <= 0:
        return float("nan")
    return rmse(t, p) / float(t.mean())


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """Forecast skill relative to a reference: ``1 - RMSE_model / RMSE_reference``.

    Positive values beat the reference, zero ties it, negative values are worse.

    Args:
        rmse_model: RMSE of the model.
        rmse_reference: RMSE of the reference (usually persistence).

    Returns:
        The skill score; NaN if the reference RMSE is zero or NaN.
    """
    if not np.isfinite(rmse_reference) or rmse_reference == 0:
        return float("nan")
    return 1.0 - rmse_model / rmse_reference
