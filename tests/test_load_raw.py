import pandas as pd
import pytest

from src.data.load_raw import COLUMNS, RAW_DIR, read_readings, read_rul


def test_read_readings_real_train_file():
    df = read_readings(RAW_DIR / "train_FD001.txt")
    assert list(df.columns) == COLUMNS
    assert len(df) == 20631
    assert df["unit_id"].nunique() == 100
    assert df["unit_id"].dtype.kind == "i" and df["cycle"].dtype.kind == "i"


def test_read_readings_wrong_column_count(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 1 0.1 0.2\n1 2 0.1 0.2\n")
    with pytest.raises(ValueError, match="столбцов"):
        read_readings(path)


def test_read_rul_numbers_units_from_one(tmp_path):
    path = tmp_path / "rul.txt"
    path.write_text("112\n98 \n69\n")
    df = read_rul(path)
    assert df.to_dict("list") == {"unit_id": [1, 2, 3], "rul": [112, 98, 69]}


def test_read_rul_real_file():
    df = read_rul(RAW_DIR / "RUL_FD001.txt")
    assert len(df) == 100
    assert (df["rul"] >= 0).all()
    assert isinstance(df, pd.DataFrame)
