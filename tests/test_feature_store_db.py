"""Интеграционные тесты Feature Store с PostgreSQL (и Redis, если задан REDIS_URL).
Пропускаются, если база недоступна или ETL ещё не выполнялся."""
import os

import fakeredis
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.data.features import FEATURE_COLS, last_row_features
from src.feature_store.materialize import materialize
from src.feature_store.online import RedisOnlineStore, needed_rows

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    try:
        from src.utils.db import get_engine
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1 FROM features.feature_registry LIMIT 1"))
            conn.execute(text("SELECT 1 FROM enriched.engine_reference_specs LIMIT 1"))
        return eng
    except Exception as exc:
        pytest.skip(f"PostgreSQL недоступен или ETL не выполнялся: {exc}")


def redis_stores():
    stores = [("fakeredis", lambda: RedisOnlineStore(fakeredis.FakeRedis(decode_responses=True), prefix="t"))]
    if os.getenv("REDIS_URL"):   # в CI — настоящий Redis
        def real():
            import redis
            r = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
            for k in r.scan_iter("pytest:*"):
                r.delete(k)
            return RedisOnlineStore(r, prefix="pytest")
        stores.append(("redis", real))
    return stores


def test_registry_table_describes_every_feature_column(engine):
    reg = pd.read_sql("SELECT feature_name FROM features.feature_registry ORDER BY position", engine)
    cols = pd.read_sql("SELECT column_name FROM information_schema.columns WHERE table_schema = 'features' "
                       "AND table_name = 'engine_features' ORDER BY ordinal_position", engine)["column_name"]
    assert reg["feature_name"].tolist() == FEATURE_COLS
    assert set(FEATURE_COLS) <= set(cols)


def test_reference_specs_loaded(engine):
    df = pd.read_sql("SELECT * FROM enriched.engine_reference_specs", engine)
    assert len(df) >= 5 and df["max_thrust_kn"].notna().all()


def test_sql_functions_backfill_and_as_of(engine):
    """SQL-функции офлайн-хранилища: пересчёт истории двигателя и выборка «на момент цикла»."""
    before = pd.read_sql("SELECT * FROM features.engine_features WHERE dataset='test' AND unit_id=34 ORDER BY cycle",
                         engine)
    with engine.begin() as conn:
        n = conn.execute(text("SELECT features.backfill_engine_features('test', ARRAY[34])")).scalar_one()
    after = pd.read_sql("SELECT * FROM features.engine_features WHERE dataset='test' AND unit_id=34 ORDER BY cycle",
                        engine)
    assert n == len(before) == len(after)
    pd.testing.assert_frame_equal(before, after)
    as_of = pd.read_sql("SELECT cycle, s_11 FROM features.get_engine_features('test', 34, 100)", engine)
    assert as_of["cycle"].tolist() == [100]
    assert as_of["s_11"].item() == before.loc[before["cycle"] == 100, "s_11"].item()


def test_python_client(engine):
    from src.feature_store.client import FeatureStore
    fs = FeatureStore(engine, RedisOnlineStore(fakeredis.FakeRedis(decode_responses=True), prefix="client"))
    assert len(fs.list_features()) == len(FEATURE_COLS)
    hist = fs.get_historical_features("train", [1, 2], columns=["s_11", "s_11_slope_20"])
    assert set(hist["unit_id"]) == {1, 2} and list(hist.columns) == ["dataset", "unit_id", "cycle", "s_11", "s_11_slope_20"]
    assert fs.get_features_as_of("test", 34, 50)["cycle"] == 50
    assert fs.get_features_as_of("test", 999_999) is None
    assert fs.backfill("test", [34]) > 0
    assert fs.get_online_features([34]) == {34: None}
    fs.materialize("test", [34])
    rec = fs.get_online_features([34])[34]
    assert rec["source"] == "offline:test" and len(rec["features"]) == len(FEATURE_COLS)


@pytest.mark.parametrize("name,make", redis_stores(), ids=[n for n, _ in redis_stores()])
def test_materialized_online_features_equal_offline(engine, name, make):
    """После материализации онлайн-признаки (по истории в Redis) совпадают с офлайн-признаками PostgreSQL."""
    store = make()
    units = [1, 34, 100]
    res = materialize(engine, store, dataset="test", units=units)
    assert res["engines"] == 3
    offline = pd.read_sql(text("""SELECT DISTINCT ON (unit_id) * FROM features.engine_features
                                  WHERE dataset = 'test' AND unit_id = ANY(:u) ORDER BY unit_id, cycle DESC"""),
                          engine, params={"u": units}).set_index("unit_id")
    for u in units:
        online = last_row_features(needed_rows(store.history(u)))
        np.testing.assert_allclose([online[c] for c in FEATURE_COLS], offline.loc[u, FEATURE_COLS].to_numpy(float),
                                   rtol=1e-7, atol=1e-7)
        rec = store.get_features(u)
        assert rec["source"] == "offline:test" and rec["cycle"] == offline.loc[u, "cycle"]
        assert store.last_cycle(u) == offline.loc[u, "cycle"]
