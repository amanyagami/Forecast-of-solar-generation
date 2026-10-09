import numpy as np
import pandas as pd
import pytest

from solarcast.evaluation import HORIZONS, run_protocol, score
from solarcast.features import build_dataset, make_features, target
from solarcast.models import ChronosModel, LightGBMModel

from .conftest import make_frame


def test_features_do_not_use_the_future() -> None:
    frame = make_frame(40)
    ds = build_dataset(frame)
    base = make_features(ds, 2)
    altered = frame.copy()
    altered.loc["2019-01-30":] = 123.0
    ds2 = build_dataset(altered)
    alt = make_features(ds2, 2)
    cut = pd.Timestamp("2019-01-29 23:00")
    # Only the centered envelope (labels) may change; causal features must not.
    pd.testing.assert_frame_equal(base.loc[:cut], alt.loc[:cut])


def test_target_is_shifted_generation() -> None:
    ds = build_dataset(make_frame(10))
    t = target(ds, 3)
    assert t.iloc[0] == ds.y.iloc[3]
    assert np.isnan(t.iloc[-1])


def test_holdout_end_to_end_lightgbm() -> None:
    ds = build_dataset(make_frame(80))
    preds = run_protocol(
        ds, "holdout", [LightGBMModel(n_estimators=150, learning_rate=0.1)], HORIZONS
    )
    assert set(preds["model"]) == {
        "persistence",
        "smart_persistence",
        "seasonal_naive",
        "lightgbm",
    }
    test_start = preds["target_time"].min()
    # nothing evaluated before the chronological test block, i.e. last 20% of days
    assert test_start >= ds.frame.index[int(len(ds.frame) * 0.8) - 96]
    metrics = score(preds, ds)
    assert {"rmse", "skill_vs_persistence", "regime", "horizon_min"} <= set(metrics.columns)
    row = metrics[(metrics["model"] == "persistence") & (metrics["regime"] == "all")][
        "skill_vs_persistence"
    ]
    assert np.allclose(row, 0.0)
    lgb = metrics[(metrics["model"] == "lightgbm") & (metrics["regime"] == "all")]
    pers = metrics[(metrics["model"] == "persistence") & (metrics["regime"] == "all")]
    # At the longest horizon a learned model must beat plain persistence on this smooth signal.
    assert lgb.set_index("horizon_min")["rmse"][60] < pers.set_index("horizon_min")["rmse"][60]


def test_unknown_protocol() -> None:
    ds = build_dataset(make_frame(10))
    with pytest.raises(ValueError):
        run_protocol(ds, "bogus", [])


class _Broken:
    name = "broken"

    def fit_predict(self, *args, **kwargs):
        raise ImportError("optional dependency missing")


def test_missing_optional_model_is_skipped() -> None:
    ds = build_dataset(make_frame(80))
    preds = run_protocol(ds, "holdout", [_Broken()], (1,))
    assert "broken" not in set(preds["model"])


def test_chronos_without_weights_raises_importable_error() -> None:
    ds = build_dataset(make_frame(30))
    model = ChronosModel(model_id="does-not-exist/none")
    with pytest.raises((ImportError, OSError, ValueError)):
        model.fit_predict(ds, (1,), ds.frame.index[-1], ds.frame.index[-96], ds.frame.index[-1])


def test_lstm_trains_and_forecasts_without_peeking() -> None:
    pytest.importorskip("torch")
    from solarcast.models import LSTMModel

    ds = build_dataset(make_frame(40))
    fit_end = pd.Timestamp("2019-01-31")
    out = LSTMModel(epochs=2, hidden=8).fit_predict(
        ds, (1, 4), fit_end, pd.Timestamp("2019-01-31"), pd.Timestamp("2019-02-05")
    )
    assert set(out) == {1, 4}
    assert (out[1] >= 0).all() and out[1].notna().all()
    assert out[1].index.min() >= pd.Timestamp("2019-01-30 23:30")
