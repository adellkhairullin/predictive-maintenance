"""Подключение к PostgreSQL по параметрам из файла .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = PROJECT_ROOT / "sql"


def get_engine() -> Engine:
    """Создаёт подключение к базе. Параметры берутся из переменных окружения или из .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    url = URL.create(
        "postgresql+psycopg2",
        username=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        database=os.environ["PGDATABASE"],
    )
    return create_engine(url)


def run_sql_file(engine: Engine, path: Path) -> None:
    """Выполняет SQL-скрипт целиком в одной транзакции."""
    sql = Path(path).read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)
