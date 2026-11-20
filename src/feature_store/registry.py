"""Реестр признаков: описание каждого признака офлайн-хранилища (features.engine_features).

Реестр строится из тех же констант, что и расчёт признаков (src/data/features.py), поэтому описание
не расходится с кодом. Описания датчиков — по документации C-MAPSS (Saxena et al., 2008).
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.features import (BASELINE_CYCLES, FEATURE_COLS, FEATURE_SENSORS, MEAN_WINDOW,
                               SLOPE_WINDOW)

FEATURE_SET_VERSION = "1"
ENTITY = "unit_id"
OFFLINE_TABLE = "features.engine_features"

# датчик: (обозначение, описание, единица)
SENSORS = {
    "s_2": ("T24", "Полная температура на выходе компрессора низкого давления", "°R"),
    "s_3": ("T30", "Полная температура на выходе компрессора высокого давления", "°R"),
    "s_4": ("T50", "Полная температура на выходе турбины низкого давления", "°R"),
    "s_7": ("P30", "Полное давление на выходе компрессора высокого давления", "psia"),
    "s_8": ("Nf", "Физическая частота вращения вентилятора", "rpm"),
    "s_9": ("Nc", "Физическая частота вращения ядра", "rpm"),
    "s_11": ("Ps30", "Статическое давление на выходе компрессора высокого давления", "psia"),
    "s_12": ("phi", "Отношение расхода топлива к Ps30", "pps/psi"),
    "s_13": ("NRf", "Приведённая частота вращения вентилятора", "rpm"),
    "s_14": ("NRc", "Приведённая частота вращения ядра", "rpm"),
    "s_15": ("BPR", "Степень двухконтурности", "—"),
    "s_17": ("htBleed", "Энтальпия отбора воздуха", "—"),
    "s_20": ("W31", "Расход охлаждающего воздуха турбины высокого давления", "lbm/s"),
    "s_21": ("W32", "Расход охлаждающего воздуха турбины низкого давления", "lbm/s"),
}

TRANSFORMS = {
    "value": ("текущее значение", None),
    f"mean_{MEAN_WINDOW}": (f"среднее за последние {MEAN_WINDOW} циклов", MEAN_WINDOW),
    f"std_{MEAN_WINDOW}": (f"стандартное отклонение за последние {MEAN_WINDOW} циклов", MEAN_WINDOW),
    f"slope_{SLOPE_WINDOW}": (f"наклон линейного тренда за последние {SLOPE_WINDOW} циклов", SLOPE_WINDOW),
    "delta_base": (f"отклонение среднего за {MEAN_WINDOW} циклов от среднего за первые {BASELINE_CYCLES} "
                   f"циклов двигателя («нормы»)", MEAN_WINDOW),
}


def build_registry() -> pd.DataFrame:
    rows = [{"feature_name": "cycle", "sensor": None, "sensor_code": None, "unit": "цикл",
             "transform": "value", "window_cycles": None,
             "description": "Номер цикла работы двигателя (возраст)"}]
    for s in FEATURE_SENSORS:
        code, desc, unit = SENSORS[s]
        for suffix, (tdesc, window) in TRANSFORMS.items():
            name = s if suffix == "value" else f"{s}_{suffix}"
            rows.append({"feature_name": name, "sensor": s, "sensor_code": code, "unit": unit,
                         "transform": suffix, "window_cycles": window,
                         "description": f"{desc} ({code}): {tdesc}"})
    df = pd.DataFrame(rows)
    assert df["feature_name"].tolist() == FEATURE_COLS, "реестр разошёлся со списком признаков"
    df.insert(0, "position", range(1, len(df) + 1))
    df["entity"] = ENTITY
    df["source_table"] = OFFLINE_TABLE
    df["feature_set_version"] = FEATURE_SET_VERSION
    df["dtype"] = "double precision"
    df.loc[df["feature_name"] == "cycle", "dtype"] = "integer"
    return df


def sync_registry(engine: Engine) -> int:
    """Перезаписывает features.feature_registry. Возвращает число признаков."""
    df = build_registry()
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS features"))
        df.to_sql("feature_registry", conn, schema="features", if_exists="replace", index=False)
        conn.execute(text("ALTER TABLE features.feature_registry ADD PRIMARY KEY (feature_name)"))
    return len(df)
