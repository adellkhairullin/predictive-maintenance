import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import MEASURE_COLS, clean_readings, clean_rul, find_low_variance_columns


def make_readings(rows):
    """rows: список (dataset, unit_id, cycle, id); все показания заполняются случайными числами."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rows, columns=["dataset", "unit_id", "cycle", "id"])
    for c in MEASURE_COLS:
        df[c] = rng.normal(size=len(df))
    return df


def test_find_low_variance_columns():
    df = make_readings([("train", 1, i, i) for i in range(1, 6)])
    df["s_1"] = 518.67              # константа
    df["s_6"] = [21.6, 21.61] * 2 + [21.6]  # два значения
    assert find_low_variance_columns(df) == ["s_1", "s_6"]


def test_clean_readings_removes_bad_rows_and_duplicates():
    df = make_readings([
        ("train", 1, 1, 1),
        ("train", 1, 2, 2),
        ("train", 1, 2, 3),   # дубль ключа: остаться должна эта запись (id=3)
        ("train", 1, 0, 4),   # цикл < 1
        ("train", 1, 3, 5),   # будет NaN в показаниях
    ])
    df.loc[df["id"] == 5, "s_2"] = np.nan
    marker = df.loc[df["id"] == 3, "s_3"].item()

    out, report = clean_readings(df, drop_cols=["s_1"])

    assert report == {"rows_in": 5, "dropped_bad_key": 1, "dropped_bad_values": 1,
                      "dropped_duplicates": 1, "rows_out": 2}
    assert out["cycle"].tolist() == [1, 2]
    assert out.loc[out["cycle"] == 2, "s_3"].item() == marker
    assert "s_1" not in out.columns and "id" not in out.columns


def test_clean_readings_nan_in_dropped_column_is_ignored():
    df = make_readings([("train", 1, 1, 1)])
    df["s_1"] = np.nan
    out, report = clean_readings(df, drop_cols=["s_1"])
    assert report["rows_out"] == 1


def test_clean_rul_validates_units():
    rul = pd.DataFrame({"id": [1, 2, 3, 4], "unit_id": [1, 2, 2, 3], "rul": [10, 5, 7, -1]})
    out = clean_rul(rul, test_units={1, 2})
    assert out.to_dict("list") == {"unit_id": [1, 2], "rul": [10, 7]}
    with pytest.raises(ValueError, match="Нет истинного RUL"):
        clean_rul(rul, test_units={1, 2, 3})
