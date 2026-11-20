"""Загрузка обучающих и тестовых данных из слоя features."""
from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from src.data.features import FEATURE_COLS
from src.utils.db import get_engine


def load_features(engine: Engine | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (train, test) из features.engine_features."""
    engine = engine or get_engine()
    df = pd.read_sql("SELECT * FROM features.engine_features WHERE dataset IN ('train', 'test') "
                     "ORDER BY dataset, unit_id, cycle", engine)
    train = df[df["dataset"] == "train"].reset_index(drop=True)
    test = df[df["dataset"] == "test"].reset_index(drop=True)
    return train, test


def split_xy(df: pd.DataFrame, cols: list[str] = FEATURE_COLS):
    """Признаки X, целевая переменная y и группы (номер двигателя) для кросс-валидации."""
    return df[cols], df["target"].astype(int), df["unit_id"]
