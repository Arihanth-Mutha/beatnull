import math

import numpy as np
import pytest

from beatnull.data import Columns, DataError, from_rows, load


def rows(*items):
    keys = ("block", "condition", "value", "position")
    return [dict(zip(keys, item)) for item in items]


def test_missing_cells_stay_missing():
    ds = from_rows(rows(("b1", "real", "1", 0), ("b1", "real", "", 0), ("b1", "real", "NA", 0),
                        ("b2", "zero", "0", 1)))
    b = ds.blocks
    assert b.value[0] == 1.0  # mean of measured trials only, not (1 + 0 + 0) / 3
    assert b.n_missing[0] == 2
    assert b.n_measured[0] == 1
    assert ds.trial_level


def test_block_with_no_measurement_is_nan():
    ds = from_rows(rows(("b1", "real", "", 0), ("b2", "zero", "1", 1)))
    assert math.isnan(ds.blocks.value[0])


def test_one_row_per_block_is_not_trial_level():
    ds = from_rows(rows(("b1", "real", "1", 0), ("b2", "zero", "0", 1)))
    assert not ds.trial_level


def test_block_cannot_change_condition():
    with pytest.raises(DataError, match="appears under conditions"):
        from_rows(rows(("b1", "real", "1", 0), ("b1", "zero", "1", 0)))


def test_block_cannot_change_position():
    with pytest.raises(DataError, match="more than one"):
        from_rows(rows(("b1", "real", "1", 0), ("b1", "real", "1", 1)))


def test_non_numeric_metric_names_the_row():
    with pytest.raises(DataError, match="non-numeric"):
        from_rows(rows(("b1", "real", "high", 0)))


def test_missing_metric_column_lists_alternatives():
    with pytest.raises(DataError, match="available columns"):
        from_rows(rows(("b1", "real", "1", 0)), Columns(metric="score"))


def test_groups_namespace_block_ids():
    data = [
        {"group": "s1", "block": "b1", "condition": "real", "value": 1},
        {"group": "s2", "block": "b1", "condition": "zero", "value": 0},
    ]
    ds = from_rows(data)
    assert ds.blocks.key == ("s1/b1", "s2/b1")


def test_csv_round_trip_and_hash(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("block,condition,value\nb1,real,1\nb2,zero,0\n")
    ds = load(path)
    assert len(ds.sha256) == 64
    np.testing.assert_array_equal(ds.blocks.value, [1.0, 0.0])
    assert ds.columns.position is None  # absent optional columns resolve to None


def test_jsonl(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"block":"b1","condition":"real","value":1}\n'
                    '{"block":"b2","condition":"zero","value":null}\n')
    ds = load(path)
    assert math.isnan(ds.blocks.value[1])
