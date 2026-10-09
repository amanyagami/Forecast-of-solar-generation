import numpy as np
import pandas as pd
import pytest

from solarcast.splits import chronological_split, rolling_origin_splits


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.date_range("2019-01-01", periods=200 * 96, freq="15min")


def test_chronological_no_test_precedes_train(index) -> None:
    sp = chronological_split(index, 0.6, 0.2, gap=4)
    assert index[sp.test].min() > index[sp.train].max()
    assert index[sp.test].min() > index[sp.val].max()
    assert index[sp.val].min() > index[sp.train].max()
    assert sp.test.min() > sp.train.max()


def test_chronological_gap_removes_overlapping_targets(index) -> None:
    sp = chronological_split(index, 0.6, 0.2, gap=4)
    boundary = index[sp.val[0]]
    assert index[sp.train].max() <= boundary - 4 * pd.Timedelta("15min") - pd.Timedelta(
        "15min"
    ) + pd.Timedelta("15min")
    assert index[sp.train].max() < boundary - 3 * pd.Timedelta("15min")


def test_chronological_covers_blocks_disjointly(index) -> None:
    sp = chronological_split(index)
    all_pos = np.concatenate([sp.train, sp.val, sp.test])
    assert len(np.unique(all_pos)) == len(all_pos)
    assert np.all(np.diff(sp.test) == 1)
    assert sp.test[-1] == len(index) - 1


def test_blocks_are_whole_days(index) -> None:
    sp = chronological_split(index)
    assert index[sp.test[0]].time() == pd.Timestamp("00:00").time()


def test_rolling_origin_expanding_and_leak_free(index) -> None:
    folds = list(rolling_origin_splits(index, n_folds=3, test_days=20, min_train_days=60, gap=4))
    assert len(folds) == 3
    prev_train = 0
    prev_test_end = -1
    for f in folds:
        assert index[f.test].min() > index[f.train].max()
        assert len(f.train) > prev_train
        assert f.test.min() > prev_test_end
        prev_train, prev_test_end = len(f.train), f.test.max()
    assert folds[-1].test[-1] == len(index) - 1


def test_rolling_too_short_raises(index) -> None:
    with pytest.raises(ValueError):
        list(rolling_origin_splits(index, n_folds=10, test_days=30, min_train_days=60))


def test_unsorted_index_rejected(index) -> None:
    shuffled = index[::-1]
    with pytest.raises(ValueError):
        chronological_split(shuffled)
    with pytest.raises(ValueError):
        list(rolling_origin_splits(shuffled))


def test_bad_fractions(index) -> None:
    with pytest.raises(ValueError):
        chronological_split(index, 0.9, 0.2)
