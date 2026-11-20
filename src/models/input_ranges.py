"""Диапазоны показаний датчиков на обучающей выборке (min/max).

API помечает значения вне этих диапазонов предупреждением — простой контроль дрейфа входных данных.

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m src.models.input_ranges
"""
from __future__ import annotations

import json

from src.data.features import FEATURE_SENSORS
from src.data.load_raw import RAW_DIR, read_readings
from src.models.registry import MODELS_DIR

RANGES_PATH = MODELS_DIR / "input_ranges.json"


def build_ranges() -> dict:
    train = read_readings(RAW_DIR / "train_FD001.txt")
    return {s: {"min": float(train[s].min()), "max": float(train[s].max())} for s in FEATURE_SENSORS}


def load_ranges() -> dict:
    return json.loads(RANGES_PATH.read_text(encoding="utf-8")) if RANGES_PATH.exists() else {}


if __name__ == "__main__":
    RANGES_PATH.write_text(json.dumps(build_ranges(), indent=2), encoding="utf-8")
    print(f"Записан {RANGES_PATH}")
