"""Интеграционные тесты с PostgreSQL. Пропускаются, если база недоступна или слои не построены
(сначала выполните python -m src.etl_pipeline)."""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.data.features import FEATURE_COLS, FEATURE_SENSORS, KEY, compute_features, make_target

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    try:
        from src.utils.db import get_engine
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1 FROM features.engine_features LIMIT 1"))
        return eng
    except Exception as exc:  # нет базы, нет .env или слои не построены
        pytest.skip(f"PostgreSQL недоступен: {exc}")


UNITS = [1, 2, 37, 100]


def test_sql_and_pandas_features_match(engine):
    """Признаки из PostgreSQL совпадают с расчётом на pandas (нет расхождения между обучением и сервисом)."""
    params = {"u": UNITS}
    cleaned = pd.read_sql(text("SELECT * FROM cleaned.sensor_readings WHERE unit_id = ANY(:u)"),
                          engine, params=params)
    db = pd.read_sql(text("SELECT * FROM features.engine_features WHERE unit_id = ANY(:u)"),
                     engine, params=params).sort_values(KEY).reset_index(drop=True)
    py = compute_features(cleaned[KEY + FEATURE_SENSORS])
    assert len(db) == len(py) > 0
    pd.testing.assert_frame_equal(db[KEY], py[KEY], check_dtype=False)
    np.testing.assert_allclose(db[FEATURE_COLS].to_numpy(float), py[FEATURE_COLS].to_numpy(float),
                               rtol=1e-7, atol=1e-7)


def test_target_matches_rul(engine):
    df = pd.read_sql("SELECT rul, target FROM features.engine_features WHERE dataset IN ('train','test')",
                     engine)
    assert (make_target(df["rul"]) == df["target"]).all()


def test_rul_is_zero_at_failure_for_train(engine):
    df = pd.read_sql("SELECT rul FROM enriched.sensor_readings WHERE dataset = 'train' AND is_last_cycle",
                     engine)
    assert len(df) == 100 and (df["rul"] == 0).all()


def test_test_rul_on_last_cycle_equals_file(engine):
    df = pd.read_sql("""SELECT e.rul AS rul_layer, r.rul AS rul_file
                        FROM enriched.sensor_readings e JOIN cleaned.rul_test r USING (unit_id)
                        WHERE e.dataset = 'test' AND e.is_last_cycle""", engine)
    assert len(df) == 100 and (df["rul_layer"] == df["rul_file"]).all()
