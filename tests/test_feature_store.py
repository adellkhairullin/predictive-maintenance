"""Тесты Feature Store: онлайн-хранилища (память и Redis через fakeredis), реестр признаков."""
import threading

import fakeredis
import numpy as np
import pytest

from src.data.features import BASELINE_CYCLES, FEATURE_COLS, FEATURE_SENSORS, SLOPE_WINDOW, compute_features
from src.feature_store.online import (CycleConflict, InMemoryOnlineStore, RedisOnlineStore,
                                      make_online_store, needed_rows)
from src.feature_store.registry import SENSORS, build_registry


def row(cycle, value=1.0):
    return {"cycle": cycle, **{s: value + cycle for s in FEATURE_SENSORS}}


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return InMemoryOnlineStore()
    return RedisOnlineStore(fakeredis.FakeRedis(decode_responses=True), prefix="test")


def test_append_and_history_keeps_only_needed_rows(store):
    last = None
    for c in range(1, 41):
        store.append(7, row(c), expected_last=last)
        last = c
    hist = store.history(7)
    cycles = [h["cycle"] for h in hist]
    assert cycles == list(range(1, BASELINE_CYCLES + 1)) + list(range(41 - SLOPE_WINDOW, 41))
    assert store.last_cycle(7) == 40 and store.count() == 1


def test_append_rejects_gap_duplicate_and_stale_expectation(store):
    store.append(1, row(1), expected_last=None)
    with pytest.raises(CycleConflict) as e:
        store.append(1, row(3), expected_last=1)           # пропуск
    assert e.value.expected == 2
    with pytest.raises(CycleConflict):
        store.append(1, row(1), expected_last=None)         # повтор
    store.append(1, row(2), expected_last=1)
    with pytest.raises(CycleConflict):
        store.append(1, row(3), expected_last=1)            # устаревшее ожидание: цикл 2 уже принят
    assert store.last_cycle(1) == 2


def test_concurrent_appends_accept_cycle_once(store):
    """Два «одновременных» запроса с одним циклом: принимается ровно один."""
    store.append(5, row(1), expected_last=None)
    results = []

    def worker():
        try:
            store.append(5, row(2), expected_last=1)
            results.append("ok")
        except CycleConflict:
            results.append("conflict")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == ["conflict"] * 7 + ["ok"]
    assert [h["cycle"] for h in store.history(5)] == [1, 2]


def test_replace_history_and_features(store):
    store.replace_history(9, [row(c) for c in range(50, 0, -1)])   # порядок не важен
    assert store.last_cycle(9) == 50
    assert len(store.history(9)) == BASELINE_CYCLES + SLOPE_WINDOW
    store.append(9, row(51), expected_last=50)                       # продолжение после материализации
    store.put_features(9, {"cycle": 51, "features": {"a": 1.0}})
    assert store.get_features(9) == {"cycle": 51, "features": {"a": 1.0}}
    assert store.reset(9) is True and store.reset(9) is False
    assert store.get_features(9) is None and store.history(9) == [] and store.count() == 0


def test_redis_keys_layout():
    r = fakeredis.FakeRedis(decode_responses=True)
    s = RedisOnlineStore(r, prefix="pdm")
    s.append(3, row(1), expected_last=None)
    assert set(r.keys("pdm:*")) == {"pdm:engine:3:base", "pdm:engine:3:recent", "pdm:engine:3:last", "pdm:engines"}


def test_online_history_gives_same_features_as_batch():
    """Признаки по истории из онлайн-хранилища совпадают с пакетным расчётом по полной истории."""
    import pandas as pd
    from src.data.features import last_row_features
    rng = np.random.default_rng(42)
    full = [{"cycle": c, **{s: float(rng.normal(100, 1)) for s in FEATURE_SENSORS}} for c in range(1, 61)]
    s = RedisOnlineStore(fakeredis.FakeRedis(decode_responses=True))
    s.replace_history(1, full)
    online = last_row_features(needed_rows(s.history(1)))
    batch = compute_features(pd.DataFrame(full).assign(dataset="x", unit_id=1)).iloc[-1]
    np.testing.assert_allclose([online[c] for c in FEATURE_COLS], batch[FEATURE_COLS].to_numpy(float), rtol=1e-9)


def test_make_online_store():
    assert make_online_store("memory").name == "memory"
    with pytest.raises(ValueError):
        make_online_store("unknown")


def test_registry_matches_feature_list():
    reg = build_registry()
    assert reg["feature_name"].tolist() == FEATURE_COLS
    assert reg["feature_name"].is_unique and reg["description"].notna().all()
    assert set(reg["sensor"].dropna()) == set(FEATURE_SENSORS) == set(SENSORS)
    assert reg.loc[reg["feature_name"] == "s_11_slope_20", "window_cycles"].item() == 20
