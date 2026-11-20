"""Загрузка исходных файлов NASA C-MAPSS FD001 в слой raw.

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m src.data.load_raw
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.db import PROJECT_ROOT, SQL_DIR, get_engine, run_sql_file

RAW_DIR = PROJECT_ROOT / "data" / "raw"

SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle", *SETTING_COLS, *SENSOR_COLS]

FILES = {"train": "train_FD001.txt", "test": "test_FD001.txt"}
RUL_FILE = "RUL_FD001.txt"


def read_readings(path: Path) -> pd.DataFrame:
    """Читает файл показаний: без заголовка, разделитель — пробелы, 26 столбцов."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: ожидалось {len(COLUMNS)} столбцов, получено {df.shape[1]}")
    df.columns = COLUMNS
    df[["unit_id", "cycle"]] = df[["unit_id", "cycle"]].astype(int)
    return df


def read_rul(path: Path) -> pd.DataFrame:
    """Читает файл с истинным RUL тестовых двигателей: строка i соответствует двигателю i."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 1:
        raise ValueError(f"{path.name}: ожидался 1 столбец, получено {df.shape[1]}")
    df.columns = ["rul"]
    df.insert(0, "unit_id", range(1, len(df) + 1))
    return df.astype(int)


def load_raw(engine: Engine, raw_dir: Path = RAW_DIR) -> dict[str, int]:
    """Создаёт схемы и полностью перезагружает данные из файлов в raw.

    Повторный запуск не создаёт дублей: строки, загруженные из этих же файлов ранее,
    удаляются в той же транзакции. Данные из потока (dataset='stream') не трогаются.
    """
    run_sql_file(engine, SQL_DIR / "schema.sql")

    frames = {}
    for dataset, name in FILES.items():
        df = read_readings(raw_dir / name)
        df.insert(0, "dataset", dataset)
        df["source"] = name
        frames[dataset] = df
    rul = read_rul(raw_dir / RUL_FILE)
    rul["source"] = RUL_FILE

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw.sensor_readings WHERE dataset IN ('train', 'test')"))
        conn.execute(text("DELETE FROM raw.rul_test"))
        for df in frames.values():
            df.to_sql("sensor_readings", conn, schema="raw", if_exists="append",
                      index=False, method="multi", chunksize=1000)
        rul.to_sql("rul_test", conn, schema="raw", if_exists="append", index=False)

    with engine.connect() as conn:
        counts = dict(conn.execute(text(
            "SELECT dataset, count(*) FROM raw.sensor_readings GROUP BY dataset")).all())
        counts["rul_test"] = conn.execute(text("SELECT count(*) FROM raw.rul_test")).scalar_one()
    return counts


if __name__ == "__main__":
    result = load_raw(get_engine())
    for table, n in result.items():
        print(f"{table}: {n} строк")
