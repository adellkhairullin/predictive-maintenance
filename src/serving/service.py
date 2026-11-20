"""Обработка одного показания двигателя: история -> признаки -> прогноз -> онлайн-хранилище.

Используется и REST API (api/main.py), и потребителем Kafka (streaming/consumer.py), поэтому
прогноз по одному и тому же циклу одинаков независимо от того, как пришли данные.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from src.data.features import FEATURE_SENSORS, last_row_features
from src.feature_store.online import CycleConflict, OnlineStore, needed_rows
from src.feature_store.registry import FEATURE_SET_VERSION
from src.models.predict import FailurePredictor

SHORT_HISTORY_CYCLES = 6   # на циклах 1–5 тренд неустойчив (см. notebooks/04_final_model.ipynb)


class PredictionService:
    def __init__(self, predictor: FailurePredictor, store: OnlineStore, ranges: dict | None = None):
        self.predictor, self.store, self.ranges = predictor, store, ranges or {}

    def warm_up(self) -> None:
        """Пробное предсказание: первый вызов догружает библиотеки (~2 с)."""
        row = {"cycle": 1, **{s: 1.0 for s in FEATURE_SENSORS}}
        self._proba(last_row_features([row]))

    def _proba(self, features: dict) -> float:
        p = self.predictor
        return float(p.model.predict_proba(pd.DataFrame([features])[p.features])[:, 1][0])

    def predict_history(self, rows: list[dict]) -> dict:
        """Прогноз для последнего цикла по готовой истории двигателя (без изменения онлайн-хранилища).

        Используется для повторно доставленных сообщений: история берётся из PostgreSQL (raw).
        """
        t0 = time.perf_counter()
        p = self.predictor
        rows = sorted(rows, key=lambda r: r["cycle"])
        history = needed_rows([{"cycle": int(r["cycle"]), **{s: float(r[s]) for s in FEATURE_SENSORS}} for r in rows])
        proba = self._proba(last_row_features(history))
        return {"cycle": int(rows[-1]["cycle"]), "probability": proba, "alarm": proba >= p.threshold,
                "threshold": p.threshold, "model_version": p.version, "history_cycles": len(history),
                "warnings": [], "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

    def process(self, reading: dict) -> dict:
        """reading — проверенные показания (unit_id, cycle, датчики). При нарушении порядка — CycleConflict."""
        t0 = time.perf_counter()
        p, st = self.predictor, self.store
        unit, cycle = int(reading["unit_id"]), int(reading["cycle"])
        row = {"cycle": cycle, **{s: float(reading[s]) for s in FEATURE_SENSORS}}

        stored = st.history(unit)
        last = stored[-1]["cycle"] if stored else None
        expected = 1 if last is None else last + 1
        if cycle != expected:
            raise CycleConflict(unit, cycle, expected)
        history = needed_rows(stored + [row])
        features = last_row_features(history)
        proba = self._proba(features)
        # атомарно: цикл принимается, только если за это время его не принял другой обработчик
        st.append(unit, row, expected_last=last)
        st.put_features(unit, {
            "cycle": cycle, "features": features, "probability": proba, "alarm": proba >= p.threshold,
            "model_version": p.version, "feature_set_version": FEATURE_SET_VERSION, "source": "online",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

        warnings = ["short_history"] if cycle < SHORT_HISTORY_CYCLES else []
        for s in FEATURE_SENSORS:
            r = self.ranges.get(s)
            if r and not (r["min"] <= row[s] <= r["max"]):
                warnings.append(f"out_of_range:{s}")
        return {"unit_id": unit, "cycle": cycle, "probability": proba, "alarm": proba >= p.threshold,
                "threshold": p.threshold, "model_version": p.version, "history_cycles": len(history),
                "warnings": warnings, "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}
