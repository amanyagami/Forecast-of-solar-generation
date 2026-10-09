"""Shared synthetic fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solarcast.data import GEN_COL, IRR_COL


def make_frame(n_days: int = 60, cloudy_every: int = 4, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic aligned frame with bell-shaped days and some cloudy days."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01 00:00", periods=n_days * 96, freq="15min")
    hour = np.asarray(idx.hour + idx.minute / 60)
    bell = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    day_no = np.asarray((idx.normalize() - idx.normalize()[0]).days)
    scale = np.where(day_no % cloudy_every == cloudy_every - 1, 0.4, 1.0)
    noise = 1 + 0.1 * rng.standard_normal(len(idx)) * (scale < 1)
    gen = 30 * bell * scale * noise
    irr = 1000 * bell * scale * noise
    return pd.DataFrame({GEN_COL: gen.clip(0), IRR_COL: irr.clip(0)}, index=idx)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Synthetic 60-day frame."""
    return make_frame()
