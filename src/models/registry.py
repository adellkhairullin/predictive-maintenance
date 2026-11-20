"""Сохранение и загрузка моделей вместе с метаданными (версия, порог, признаки, метрики, библиотеки)."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib

from src.utils.db import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
LIBS = ["scikit-learn", "pandas", "numpy", "scipy", "joblib"]


def library_versions() -> dict:
    return {"python": platform.python_version(), **{lib: version(lib) for lib in LIBS}}


def save_model(model, name: str, metadata: dict, models_dir: Path = MODELS_DIR) -> Path:
    """Сохраняет модель в models/<name>.joblib и метаданные в models/<name>.json."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    meta = {"name": name, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "libraries": library_versions(), **metadata}
    (models_dir / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    return path


def load_model(name: str, models_dir: Path = MODELS_DIR):
    """Возвращает (модель, метаданные)."""
    model = joblib.load(models_dir / f"{name}.joblib")
    meta = json.loads((models_dir / f"{name}.json").read_text(encoding="utf-8"))
    return model, meta
