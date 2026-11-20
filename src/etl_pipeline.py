"""ETL-пайплайн: файлы -> raw -> cleaned -> enriched -> features, с проверками качества.

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m src.etl_pipeline
"""
from __future__ import annotations

import logging
import sys

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.cleaning import build_cleaned
from src.data.features import FEATURE_COLS, FEATURE_SENSORS, HORIZON, LAYERS_SQL_PATH
from src.data.load_raw import load_raw
from src.utils.db import PROJECT_ROOT, get_engine, run_sql_file

log = logging.getLogger("etl")
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


class DataQualityError(RuntimeError):
    """Проверка качества данных не пройдена."""


def check_layers(engine: Engine) -> dict:
    """Проверки после построения слоёв. При нарушении выбрасывает DataQualityError."""
    q = lambda sql: pd.read_sql(text(sql), engine)
    counts = q("""
        SELECT 'cleaned' AS layer, count(*) AS n FROM cleaned.sensor_readings
        UNION ALL SELECT 'enriched', count(*) FROM enriched.sensor_readings
        UNION ALL SELECT 'features', count(*) FROM features.engine_features
    """).set_index("layer")["n"].to_dict()

    problems = []
    if len(set(counts.values())) != 1:
        problems.append(f"число строк в слоях различается: {counts}")

    null_expr = " + ".join(f"count(*) FILTER (WHERE {c} IS NULL)" for c in FEATURE_COLS)
    nulls = q(f"SELECT {null_expr} AS n FROM features.engine_features")["n"].item()
    if nulls:
        problems.append(f"пропуски в признаках: {nulls}")

    labels = q("""
        SELECT dataset, count(*) AS rows, count(DISTINCT unit_id) AS units,
               count(*) FILTER (WHERE rul IS NULL OR rul < 0) AS bad_rul,
               avg(target) AS target_share
        FROM features.engine_features
        WHERE dataset IN ('train', 'test')
        GROUP BY dataset
    """).set_index("dataset")
    if labels["bad_rul"].sum():
        problems.append(f"пустой или отрицательный RUL: {labels['bad_rul'].to_dict()}")
    train_share = float(labels.loc["train", "target_share"])
    if not 0.05 < train_share < 0.5:
        problems.append(f"подозрительная доля класса 1 в train: {train_share:.3f}")

    if problems:
        raise DataQualityError("; ".join(problems))
    return {"rows_per_layer": counts, "nulls_in_features": int(nulls),
            "labels": labels.round(4).to_dict("index")}


def export_sample(engine: Engine, units: tuple[int, ...] = (1, 2)) -> str:
    """Сохраняет небольшой пример слоя features (обучающие двигатели units) в data/samples."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / "features_sample.csv"
    df = pd.read_sql(text("SELECT * FROM features.engine_features "
                          "WHERE dataset = 'train' AND unit_id = ANY(:u) ORDER BY unit_id, cycle"),
                     engine, params={"u": list(units)})
    df.to_csv(path, index=False, float_format="%.6g")
    return str(path.relative_to(PROJECT_ROOT))


def run_pipeline(engine: Engine) -> dict:
    log.info("1/5 Загрузка файлов в raw")
    raw = load_raw(engine)
    log.info("    %s", raw)

    log.info("2/5 Очистка: raw -> cleaned")
    cleaned = build_cleaned(engine)
    missing = set(FEATURE_SENSORS) & set(cleaned["dropped_columns"])
    if missing:
        raise DataQualityError(f"датчики для признаков удалены при очистке: {sorted(missing)}")
    log.info("    строк %s, удалено столбцов %s", cleaned["rows_out"], len(cleaned["dropped_columns"]))

    log.info("3/5 Слои enriched и features (%s)", LAYERS_SQL_PATH.name)
    run_sql_file(engine, LAYERS_SQL_PATH)

    log.info("4/5 Проверки качества")
    checks = check_layers(engine)
    log.info("    %s", checks)

    log.info("5/5 Пример данных")
    sample = export_sample(engine)
    log.info("    %s", sample)
    return {"raw": raw, "cleaned": cleaned, "checks": checks, "sample": sample,
            "horizon": HORIZON, "n_features": len(FEATURE_COLS)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    run_pipeline(get_engine())
    log.info("Готово")
