import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.feature_store.online import InMemoryOnlineStore, RedisOnlineStore
from api.schemas import EXAMPLE
from src.data.features import FEATURE_SENSORS, compute_features
from src.data.load_raw import RAW_DIR, read_readings
from src.models.input_ranges import build_ranges
from src.models.predict import FailurePredictor


class MemoryLogger:
    name = "memory"

    def __init__(self):
        self.records = []

    def log(self, record):
        self.records.append(record)

    def close(self):
        pass


@pytest.fixture(scope="module")
def predictor():
    return FailurePredictor("final_v2")


@pytest.fixture(scope="module")
def test_readings():
    return read_readings(RAW_DIR / "test_FD001.txt")


@pytest.fixture
def logger():
    return MemoryLogger()


def make_store(kind):
    if kind == "memory":
        return InMemoryOnlineStore()
    import fakeredis
    return RedisOnlineStore(fakeredis.FakeRedis(decode_responses=True))


@pytest.fixture(params=["memory", "redis"])
def client(request, predictor, logger):
    """Каждый тест API выполняется с обоими онлайн-хранилищами: в памяти и Redis (fakeredis)."""
    app = create_app(predictor=predictor, store=make_store(request.param), prediction_logger=logger,
                     ranges=build_ranges())
    with TestClient(app) as c:
        c.store_kind = request.param
        yield c


def reading(unit_id=1, cycle=1, **override):
    return {**EXAMPLE, "unit_id": unit_id, "cycle": cycle, **override}


def records(df: pd.DataFrame, unit: int) -> list[dict]:
    d = df[df["unit_id"] == unit].sort_values("cycle")
    return [{k: (int(v) if k in ("unit_id", "cycle") else float(v)) for k, v in r.items()}
            for r in d.to_dict("records")]


# ---------- служебные эндпоинты ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "v2", "engines_tracked": 0,
                        "online_store": client.store_kind, "prediction_log": "memory"}


def test_model_info(client, predictor):
    r = client.get("/model").json()
    assert r["version"] == "v2" and r["n_features"] == 71 and r["threshold"] == predictor.threshold
    assert "scikit-learn" in r["libraries"]


# ---------- предсказание ----------

def test_predict_first_cycle(client, logger):
    r = client.post("/predict", json=reading())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["alarm"] == (body["probability"] >= body["threshold"])
    assert body["warnings"] == ["short_history"] and body["history_cycles"] == 1
    assert body["latency_ms"] > 0
    assert len(logger.records) == 1 and logger.records[0]["unit_id"] == 1
    assert client.get("/health").json()["engines_tracked"] == 1


def test_api_matches_batch_features_for_whole_engine(client, predictor, test_readings):
    """Последовательная подача циклов через API даёт те же вероятности, что пакетный расчёт."""
    unit = 34   # тестовый двигатель, близкий к отказу
    rows = records(test_readings, unit)
    api_proba = [client.post("/predict", json=r).json()["probability"] for r in rows]
    batch = predictor.predict(compute_features(test_readings[test_readings["unit_id"] == unit].assign(dataset="test")))
    np.testing.assert_allclose(api_proba, batch["probability"].to_numpy(), rtol=1e-6, atol=1e-9)
    last = client.post("/predict", json=reading(unit, len(rows) + 1, **{s: rows[-1][s] for s in FEATURE_SENSORS}))
    assert last.json()["history_cycles"] == 30   # 10 циклов «нормы» + 20 последних


def test_alarm_near_failure(client, test_readings):
    rows = records(test_readings, 34)   # истинный RUL на последнем цикле = 7
    for r in rows[:-1]:
        client.post("/predict", json=r)
    assert client.post("/predict", json=rows[-1]).json()["alarm"] is True


def test_out_of_range_warning(client):
    r = client.post("/predict", json=reading(s_11=60.0)).json()
    assert "out_of_range:s_11" in r["warnings"]


def test_optional_sensors_may_be_omitted(client):
    body = {k: v for k, v in reading().items() if k not in ("s_1", "s_5", "setting_3")}
    assert client.post("/predict", json=body).status_code == 200


# ---------- порядок циклов ----------

def test_cycle_must_start_from_one(client):
    r = client.post("/predict", json=reading(cycle=5))
    assert r.status_code == 409
    assert r.json()["detail"]["expected_cycle"] == 1


def test_duplicate_and_gap_are_rejected(client):
    for c in (1, 2):
        assert client.post("/predict", json=reading(cycle=c)).status_code == 200
    dup = client.post("/predict", json=reading(cycle=2))
    assert dup.status_code == 409 and "повтор" in dup.json()["detail"]["error"]
    gap = client.post("/predict", json=reading(cycle=4))
    assert gap.status_code == 409 and gap.json()["detail"]["expected_cycle"] == 3


def test_engines_are_independent(client):
    assert client.post("/predict", json=reading(unit_id=1, cycle=1)).status_code == 200
    assert client.post("/predict", json=reading(unit_id=2, cycle=1)).status_code == 200


def test_reset_engine(client):
    client.post("/predict", json=reading(cycle=1))
    client.post("/predict", json=reading(cycle=2))
    assert client.delete("/engines/1").status_code == 204
    assert client.post("/predict", json=reading(cycle=1)).status_code == 200
    assert client.delete("/engines/999").status_code == 404


# ---------- валидация ----------

@pytest.mark.parametrize("change", [
    {"s_2": None},                 # обязательный датчик пустой
    {"s_2": "abc"},                # не число
    {"s_2": -5.0},                 # отрицательное показание
    {"cycle": 0},                  # цикл должен быть > 0
    {"unit_id": -1},
    {"setting_1": 500.0},          # режимный параметр вне пределов
    {"s2": 641.8},                 # неизвестное поле (опечатка)
])
def test_validation_errors(client, change):
    r = client.post("/predict", json={**reading(), **change})
    assert r.status_code == 422


def test_missing_required_sensor(client):
    body = {k: v for k, v in reading().items() if k != "s_11"}
    r = client.post("/predict", json=body)
    assert r.status_code == 422 and "s_11" in r.text


@pytest.mark.parametrize("bad", ["NaN", "Infinity"])
def test_nan_and_infinity_rejected(client, bad):
    raw = json.dumps(reading()).replace('"s_3": 1589.7', f'"s_3": {bad}')
    r = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_rejected_request_does_not_change_history(client):
    client.post("/predict", json=reading(cycle=1))
    client.post("/predict", json=reading(cycle=2, s_2=-1.0))     # 422
    assert client.post("/predict", json=reading(cycle=2)).status_code == 200


# ---------- онлайн-хранилище признаков ----------

def test_engine_features_after_predict(client, predictor):
    assert client.get("/engines/1/features").status_code == 404
    client.post("/predict", json=reading(cycle=1))
    body = client.post("/predict", json=reading(cycle=2, s_2=642.5)).json()
    f = client.get("/engines/1/features")
    assert f.status_code == 200
    rec = f.json()
    assert rec["cycle"] == 2 and rec["source"] == "online" and rec["model_version"] == "v2"
    assert rec["probability"] == pytest.approx(body["probability"])
    assert set(rec["features"]) == set(predictor.features) and rec["features"]["s_2"] == 642.5


def test_reset_clears_online_features(client):
    client.post("/predict", json=reading(cycle=1))
    client.delete("/engines/1")
    assert client.get("/engines/1/features").status_code == 404


# ---------- журнал ----------

def test_null_logger_when_db_disabled():
    from api.prediction_log import make_logger
    assert make_logger("none").name == "none"
