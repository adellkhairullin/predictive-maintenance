"""Python API Feature Store: единая точка чтения признаков из офлайн- и онлайн-хранилища.

Пример:
    from src.feature_store.client import FeatureStore
    fs = FeatureStore.from_env()                       # PostgreSQL из .env, онлайн-хранилище по PDM_ONLINE_STORE
    fs.list_features().head()                          # реестр признаков
    fs.get_historical_features("train", [1, 2])        # история (обучение)
    fs.get_features_as_of("test", 34, as_of_cycle=100) # состояние на момент цикла
    fs.materialize("test", [34])                       # офлайн -> онлайн
    fs.get_online_features([34])                       # последние признаки для сервиса
"""
from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.feature_store.materialize import materialize
from src.feature_store.online import OnlineStore, make_online_store
from src.utils.db import get_engine


class FeatureStore:
    def __init__(self, engine: Engine, online: OnlineStore):
        self.engine, self.online = engine, online

    @classmethod
    def from_env(cls) -> "FeatureStore":
        return cls(get_engine(), make_online_store(os.getenv("PDM_ONLINE_STORE", "memory"), os.getenv("REDIS_URL")))

    # ---------- офлайн (PostgreSQL) ----------
    def list_features(self) -> pd.DataFrame:
        """Реестр признаков: имя, датчик и его смысл, преобразование, окно, версия набора."""
        return pd.read_sql("SELECT * FROM features.feature_registry ORDER BY position", self.engine)

    def get_historical_features(self, dataset: str, unit_ids: list[int] | None = None,
                                columns: list[str] | None = None) -> pd.DataFrame:
        """Все циклы выбранных двигателей из features.engine_features."""
        cols = ", ".join(["dataset", "unit_id", "cycle", *(columns or [])]) if columns else "*"
        sql = f"SELECT {cols} FROM features.engine_features WHERE dataset = :d"
        params: dict = {"d": dataset}
        if unit_ids:
            sql += " AND unit_id = ANY(:u)"
            params["u"] = list(unit_ids)
        return pd.read_sql(text(sql + " ORDER BY unit_id, cycle"), self.engine, params=params)

    def get_features_as_of(self, dataset: str, unit_id: int, as_of_cycle: int | None = None) -> dict | None:
        """Признаки двигателя на момент цикла (SQL-функция features.get_engine_features)."""
        df = pd.read_sql(text("SELECT * FROM features.get_engine_features(:d, :u, :c)"), self.engine,
                         params={"d": dataset, "u": unit_id, "c": as_of_cycle})
        return None if df.empty else df.iloc[0].to_dict()

    def backfill(self, dataset: str | None = None, unit_ids: list[int] | None = None) -> int:
        """Пересчитать исторические признаки (SQL-функция features.backfill_engine_features)."""
        with self.engine.begin() as conn:
            return conn.execute(text("SELECT features.backfill_engine_features(:d, :u)"),
                                {"d": dataset, "u": list(unit_ids) if unit_ids else None}).scalar_one()

    # ---------- онлайн (Redis / память) ----------
    def materialize(self, dataset: str = "test", unit_ids: list[int] | None = None) -> dict:
        """Перенести историю и последние признаки из PostgreSQL в онлайн-хранилище."""
        return materialize(self.engine, self.online, dataset, unit_ids)

    def get_online_features(self, unit_ids: list[int]) -> dict[int, dict | None]:
        """Последний вектор признаков (и прогноз, если был) по каждому двигателю."""
        return {u: self.online.get_features(u) for u in unit_ids}
