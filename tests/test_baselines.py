import numpy as np
import pandas as pd
import pytest

from solarcast.baselines import persistence, seasonal_naive, smart_persistence
from solarcast.data import GEN_COL
from solarcast.labeling import clear_sky_envelope

from .conftest import make_frame


def test_persistence_is_identity_at_origin() -> None:
    y = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2019-01-01", periods=3, freq="15min"))
    assert persistence(y, 2).tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        persistence(y, 0)


def test_seasonal_naive_uses_previous_day() -> None:
    y = make_frame(5)[GEN_COL]
    f = seasonal_naive(y, horizon=2)
    t = pd.Timestamp("2019-01-03 11:00")
    # forecast issued at t for t+2 steps equals the value one day before t+2 steps
    assert f.loc[t] == y.loc[t + pd.Timedelta("30min") - pd.Timedelta("1D")]
    with pytest.raises(ValueError):
        seasonal_naive(y, horizon=96)


def test_smart_persistence_beats_persistence_on_clear_days() -> None:
    y = make_frame(30, cloudy_every=1000)[GEN_COL]
    env = clear_sky_envelope(y, mode="trailing")
    h = 4
    truth = y.shift(-h)
    day = (y.index >= "2019-01-20") & (env > 1.5)
    smart = smart_persistence(y, env, h)
    err_smart = np.sqrt(np.nanmean((smart[day] - truth[day]) ** 2))
    err_pers = np.sqrt(np.nanmean((persistence(y, h)[day] - truth[day]) ** 2))
    assert err_smart < 0.5 * err_pers


def test_smart_persistence_uses_only_causal_envelope() -> None:
    y = make_frame(30)[GEN_COL]
    env = clear_sky_envelope(y, mode="trailing")
    base = smart_persistence(y, env, 2)
    y2 = y.copy()
    y2.loc["2019-01-25":] = 777.0
    env2 = clear_sky_envelope(y2, mode="trailing")
    alt = smart_persistence(y2, env2, 2)
    pd.testing.assert_series_equal(base.loc[:"2019-01-24 23:00"], alt.loc[:"2019-01-24 23:00"])
