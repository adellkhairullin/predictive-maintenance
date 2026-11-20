"""Очистка данных: переход из слоя raw в слой cleaned.

Правила выведены из анализа данных (notebooks/01_data_preparation.ipynb):
1. Строки без номера двигателя или цикла, с циклом < 1 или с нечисловыми показаниями удаляются.
2. Дубли по ключу (dataset, unit_id, cycle) удаляются, остаётся последняя загруженная запись.
3. Столбцы, которые на обучающей выборке почти не меняются (не больше 2 разных значений),
   удаляются: они не несут информации об износе. Список определяется только по train,
   чтобы сведения из тестовой выборки не влияли на обработку (нет утечки данных).
4. Выбросы НЕ удаляются: большинство из них приходится на последние циклы перед отказом,
   то есть это сигнал износа, а не ошибка измерения.

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m src.data.cleaning
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.load_raw import SENSOR_COLS, SETTING_COLS
from src.utils.db import get_engine

KEY = ["dataset", "unit_id", "cycle"]
MEASURE_COLS = SETTING_COLS + SENSOR_COLS
MAX_UNIQUE_TO_DROP = 2


def find_low_variance_columns(train: pd.DataFrame, cols: list[str] = MEASURE_COLS,
                              max_unique: int = MAX_UNIQUE_TO_DROP) -> list[str]:
    """Возвращает столбцы, у которых на обучающей выборке не больше max_unique разных значений."""
    return [c for c in cols if train[c].nunique(dropna=True) <= max_unique]


def clean_readings(df: pd.DataFrame, drop_cols: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Применяет правила очистки к показаниям. Возвращает очищенные данные и отчёт о числе удалённых строк."""
    report = {"rows_in": len(df)}
    keep_cols = [c for c in MEASURE_COLS if c not in drop_cols]

    bad_key = df[KEY].isna().any(axis=1) | (df["cycle"] < 1)
    report["dropped_bad_key"] = int(bad_key.sum())
    df = df[~bad_key]

    values = df[keep_cols].apply(pd.to_numeric, errors="coerce")
    bad_values = ~np.isfinite(values).all(axis=1)
    report["dropped_bad_values"] = int(bad_values.sum())
    df = df[~bad_values]

    order = [c for c in ["loaded_at", "id"] if c in df.columns]
    if order:
        df = df.sort_values(order)
    dup = df.duplicated(KEY, keep="last")
    report["dropped_duplicates"] = int(dup.sum())
    df = df[~dup]

    out = df[KEY + keep_cols].copy()
    out[["unit_id", "cycle"]] = out[["unit_id", "cycle"]].astype(int)
    out = out.sort_values(KEY).reset_index(drop=True)
    report["rows_out"] = len(out)
    return out, report


def clean_rul(rul: pd.DataFrame, test_units: set[int]) -> pd.DataFrame:
    """Проверяет таблицу истинного RUL: неотрицательные значения, по одной записи на каждый тестовый двигатель."""
    out = rul.dropna(subset=["unit_id", "rul"])
    out = out[out["rul"] >= 0]
    if "id" in out.columns:
        out = out.sort_values("id")
    out = out.drop_duplicates("unit_id", keep="last")[["unit_id", "rul"]].astype(int)
    missing = test_units - set(out["unit_id"])
    if missing:
        raise ValueError(f"Нет истинного RUL для тестовых двигателей: {sorted(missing)}")
    return out.sort_values("unit_id").reset_index(drop=True)


def build_cleaned(engine: Engine) -> dict:
    """Читает raw, очищает и полностью перезаписывает таблицы слоя cleaned."""
    raw = pd.read_sql("SELECT * FROM raw.sensor_readings", engine)
    rul_raw = pd.read_sql("SELECT * FROM raw.rul_test", engine)

    drop_cols = find_low_variance_columns(raw[raw["dataset"] == "train"])
    readings, report = clean_readings(raw, drop_cols)
    test_units = set(readings.loc[readings["dataset"] == "test", "unit_id"])
    rul = clean_rul(rul_raw, test_units)

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS cleaned"))
        readings.to_sql("sensor_readings", conn, schema="cleaned", if_exists="replace",
                        index=False, method="multi", chunksize=1000)
        conn.execute(text("ALTER TABLE cleaned.sensor_readings ADD PRIMARY KEY (dataset, unit_id, cycle)"))
        rul.to_sql("rul_test", conn, schema="cleaned", if_exists="replace", index=False)
        conn.execute(text("ALTER TABLE cleaned.rul_test ADD PRIMARY KEY (unit_id)"))

    report["dropped_columns"] = drop_cols
    report["rul_rows"] = len(rul)
    return report


if __name__ == "__main__":
    for key, value in build_cleaned(get_engine()).items():
        print(f"{key}: {value}")
