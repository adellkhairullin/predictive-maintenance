"""Интеграционный тест журнала предсказаний. Пропускается, если PostgreSQL недоступен."""
import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration
TEST_UNIT = 987_654   # номер, которого нет в данных


@pytest.fixture(scope="module")
def db_logger():
    try:
        from api.prediction_log import DbPredictionLogger
        from src.utils.db import get_engine
        return DbPredictionLogger(get_engine())
    except Exception as exc:
        pytest.skip(f"PostgreSQL недоступен: {exc}")


def test_prediction_is_written_to_db(db_logger):
    rec = {"unit_id": TEST_UNIT, "cycle": 1, "model_version": "test", "probability": 0.42, "alarm": False,
           "threshold": 0.97, "latency_ms": 1.5, "warnings": ["short_history"], "source": "pytest"}
    try:
        db_logger.log(rec)
        db_logger.flush()   # запись идёт в отдельном потоке — дождаться её
        with db_logger.engine.connect() as conn:
            row = conn.execute(text("SELECT probability, alarm, warnings, source FROM monitoring.predictions "
                                    "WHERE unit_id = :u"), {"u": TEST_UNIT}).one()
        assert row.probability == 0.42 and row.alarm is False
        assert row.warnings == ["short_history"] and row.source == "pytest"
    finally:
        with db_logger.engine.begin() as conn:
            conn.execute(text("DELETE FROM monitoring.predictions WHERE unit_id = :u"), {"u": TEST_UNIT})


def test_logger_survives_bad_record(db_logger):
    db_logger.log({"unit_id": None})   # ошибка БД пишется в лог, поток журнала не падает
    db_logger.flush()
    good = {"unit_id": TEST_UNIT, "cycle": 2, "model_version": "test", "probability": 0.1, "alarm": False,
            "threshold": 0.97, "latency_ms": 1.0, "warnings": [], "source": "pytest"}
    try:
        db_logger.log(good)
        db_logger.flush()
        with db_logger.engine.connect() as conn:
            n = conn.execute(text("SELECT count(*) FROM monitoring.predictions WHERE unit_id = :u"),
                             {"u": TEST_UNIT}).scalar_one()
        assert n == 1
    finally:
        with db_logger.engine.begin() as conn:
            conn.execute(text("DELETE FROM monitoring.predictions WHERE unit_id = :u"), {"u": TEST_UNIT})


def test_close_flushes_pending_records(db_logger):
    """Записи, оставшиеся в очереди, дописываются при остановке; плохая запись не губит пачку."""
    from api.prediction_log import DbPredictionLogger
    lg = DbPredictionLogger(db_logger.engine)
    try:
        lg.log({"unit_id": None})
        for c in range(1, 51):
            lg.log({"unit_id": TEST_UNIT, "cycle": c, "model_version": "test", "probability": 0.1,
                    "alarm": False, "threshold": 0.97, "latency_ms": 1.0, "warnings": [], "source": "pytest"})
        lg.close()
        with lg.engine.connect() as conn:
            n = conn.execute(text("SELECT count(*) FROM monitoring.predictions WHERE unit_id = :u"),
                             {"u": TEST_UNIT}).scalar_one()
        assert n == 50
    finally:
        with lg.engine.begin() as conn:
            conn.execute(text("DELETE FROM monitoring.predictions WHERE unit_id = :u"), {"u": TEST_UNIT})
