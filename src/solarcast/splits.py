"""Chronological and rolling-origin splits that never shuffle and never leak.

All splitters operate on a ``DatetimeIndex`` (typically the *target* timestamps of forecast
rows) and return integer positions. Boundaries fall on midnight so no day is cut in half, and
an optional ``gap`` (in time steps) removes the rows immediately before each evaluation block
from training, which is required for multi-step targets.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """Integer positions of a train/validation/test partition."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _positions(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Return positions of ``index`` entries in ``[start, end)``."""
    return np.flatnonzero((index >= start) & (index < end))


def chronological_split(
    index: pd.DatetimeIndex,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    gap: int = 0,
    freq: str = "15min",
) -> Split:
    """Split an index into consecutive train/validation/test blocks by whole days.

    Args:
        index: Sorted timestamps.
        train_frac: Fraction of days for training.
        val_frac: Fraction of days for validation; the remainder is the test block.
        gap: Number of time steps removed from the end of train and validation so that
            multi-step targets cannot overlap the following block.
        freq: Step size used to convert ``gap`` to time.

    Returns:
        A :class:`Split` with strictly increasing, non-overlapping blocks.

    Raises:
        ValueError: If the fractions are invalid or the index is not sorted.
    """
    if not (0 < train_frac and 0 <= val_frac and train_frac + val_frac < 1):
        raise ValueError("Need 0 < train_frac, 0 <= val_frac and train_frac + val_frac < 1")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending (no shuffling)")
    days = index.normalize().unique()
    n_train = int(len(days) * train_frac)
    n_val = int(len(days) * val_frac)
    t_val = days[n_train]
    t_test = days[n_train + n_val]
    step = pd.Timedelta(freq) * gap
    return Split(
        train=_positions(index, index.min(), t_val - step),
        val=_positions(index, t_val, t_test - step),
        test=_positions(index, t_test, index.max() + pd.Timedelta(freq)),
    )


def rolling_origin_splits(
    index: pd.DatetimeIndex,
    n_folds: int = 4,
    test_days: int = 30,
    min_train_days: int = 120,
    gap: int = 0,
    freq: str = "15min",
) -> Iterator[Split]:
    """Yield expanding-window rolling-origin folds.

    Fold ``k`` trains on everything before its test block (minus ``gap`` steps) and tests on the
    next ``test_days`` days; test blocks are consecutive and end at the last day of the index.

    Args:
        index: Sorted timestamps.
        n_folds: Number of folds.
        test_days: Length of each test block in days.
        min_train_days: Minimum number of days before the first test block.
        gap: Steps removed from the end of each training block.
        freq: Step size used to convert ``gap`` to time.

    Yields:
        :class:`Split` objects with an empty ``val`` array.

    Raises:
        ValueError: If the index is too short for the requested folds.
    """
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending (no shuffling)")
    days = index.normalize().unique()
    first_test = len(days) - n_folds * test_days
    if first_test < min_train_days:
        raise ValueError(
            f"{len(days)} days is too short for {n_folds} folds of {test_days} days "
            f"with at least {min_train_days} training days"
        )
    step = pd.Timedelta(freq) * gap
    stop = index.max() + pd.Timedelta(freq)
    for k in range(n_folds):
        t_start = days[first_test + k * test_days]
        t_stop = days[first_test + (k + 1) * test_days] if k < n_folds - 1 else stop
        yield Split(
            train=_positions(index, index.min(), t_start - step),
            val=np.array([], dtype=int),
            test=_positions(index, t_start, t_stop),
        )
