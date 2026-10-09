import numpy as np
import pandas as pd

from solarcast.data import GEN_COL
from solarcast.labeling import (
    CLEAR,
    CLOUDY,
    UNKNOWN,
    clear_sky_envelope,
    clearness_ratio,
    daylight_mask,
    label_days,
    label_per_step,
)

from .conftest import make_frame


def test_labels_match_construction() -> None:
    frame = make_frame(n_days=60, cloudy_every=4)
    y = frame[GEN_COL]
    labels = label_days(y, clear_sky_envelope(y))
    expected = ["cloudy" if d % 4 == 3 else "clear" for d in range(60)]
    assert (labels.to_numpy() == np.array(expected)).mean() > 0.95


def test_missing_day_is_unknown() -> None:
    frame = make_frame(n_days=30)
    y = frame[GEN_COL].copy()
    env = clear_sky_envelope(y)
    y.loc["2019-01-10"] = np.nan
    assert label_days(y, env).loc["2019-01-10"] == UNKNOWN


def test_trailing_envelope_is_causal() -> None:
    frame = make_frame(n_days=40)
    y = frame[GEN_COL]
    env = clear_sky_envelope(y, mode="trailing")
    altered = y.copy()
    altered.loc["2019-01-25":] = 999.0
    env2 = clear_sky_envelope(altered, mode="trailing")
    cut = "2019-01-24 23:45"
    pd.testing.assert_series_equal(env.loc[:cut], env2.loc[:cut])


def test_trailing_envelope_first_days_nan() -> None:
    y = make_frame(n_days=20)[GEN_COL]
    env = clear_sky_envelope(y, mode="trailing", min_days=5)
    assert env.loc["2019-01-03"].isna().all()
    assert env.loc["2019-01-15"].notna().any()


def test_clearness_ratio_and_daylight() -> None:
    y = make_frame(n_days=30, cloudy_every=1000)[GEN_COL]
    env = clear_sky_envelope(y)
    kc = clearness_ratio(y, env)
    assert kc.dropna().between(0.0, 1.5).all()
    assert abs(kc.dropna().median() - 1.0) < 0.1
    assert not daylight_mask(env).loc["2019-01-15 02:00"]
    assert daylight_mask(env).loc["2019-01-15 12:00"]


def test_label_per_step_broadcast() -> None:
    frame = make_frame(n_days=10)
    y = frame[GEN_COL]
    per_step = label_per_step(label_days(y, clear_sky_envelope(y)), frame.index)
    assert len(per_step) == len(frame)
    assert set(per_step.dropna()) <= {CLEAR, CLOUDY, UNKNOWN}


def test_invalid_mode() -> None:
    import pytest

    with pytest.raises(ValueError):
        clear_sky_envelope(make_frame(5)[GEN_COL], mode="nope")
