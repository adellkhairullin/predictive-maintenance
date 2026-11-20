"""Предсказание по сохранённой модели: вероятность отказа в горизонте 30 циклов и флаг тревоги.

Используется API и потоковой обработкой. Признаки должны быть рассчитаны так же, как при обучении
(src.data.features.compute_features).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.registry import MODELS_DIR, load_model

DEFAULT_MODEL = "final_v2"


class FailurePredictor:
    def __init__(self, name: str = DEFAULT_MODEL, models_dir: Path = MODELS_DIR):
        self.model, self.meta = load_model(name, models_dir)
        self.name = name
        self.version = self.meta.get("version", name)
        self.features: list[str] = self.meta["features"]
        self.threshold: float = float(self.meta["threshold"])

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """features — таблица с нужными столбцами-признаками; лишние столбцы игнорируются."""
        missing = [c for c in self.features if c not in features.columns]
        if missing:
            raise ValueError(f"Нет признаков: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        proba = self.model.predict_proba(features[self.features])[:, 1]
        return pd.DataFrame({"probability": proba, "alarm": proba >= self.threshold},
                            index=features.index)

    def predict_one(self, features: dict) -> dict:
        row = self.predict(pd.DataFrame([features])).iloc[0]
        return {"probability": float(row["probability"]), "alarm": bool(row["alarm"]),
                "threshold": self.threshold, "model_version": self.version}
