import numpy as np
import pytest

from solarcast.metrics import mae, nrmse, rmse, skill_score


def test_rmse_mae_known_values() -> None:
    t, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])
    assert rmse(t, p) == pytest.approx(np.sqrt(4 / 3))
    assert mae(t, p) == pytest.approx(2 / 3)


def test_nan_pairs_ignored_and_empty_is_nan() -> None:
    assert rmse([1.0, np.nan], [2.0, 5.0]) == pytest.approx(1.0)
    assert np.isnan(rmse([np.nan], [1.0]))
    assert np.isnan(mae([], []))


def test_nrmse() -> None:
    assert nrmse([2.0, 4.0], [3.0, 5.0]) == pytest.approx(1.0 / 3.0)
    assert np.isnan(nrmse([0.0, 0.0], [1.0, 1.0]))


def test_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])


def test_skill_score() -> None:
    assert skill_score(1.0, 2.0) == pytest.approx(0.5)
    assert skill_score(2.0, 2.0) == 0.0
    assert skill_score(3.0, 2.0) < 0
    assert np.isnan(skill_score(1.0, 0.0))
