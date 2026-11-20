"""Материализация: перенос данных из офлайн-хранилища (PostgreSQL) в онлайн-хранилище (Redis).

Для каждого двигателя выбранного набора в Redis записываются:
* история, нужная для онлайн-расчёта признаков (первые 10 и последние 20 циклов из cleaned);
* последний вектор признаков из features.engine_features.
После этого API продолжает расчёт с того места, где закончились данные: следующий цикл двигателя
можно сразу отправлять в /predict.

Запуск из корня проекта (нужен Redis):
    .\\.venv\\Scripts\\python.exe -m src.feature_store.materialize --dataset test
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.features import FEATURE_COLS, FEATURE_SENSORS
from src.feature_store.online import OnlineStore, make_online_store
from src.feature_store.registry import FEATURE_SET_VERSION
from src.utils.db import get_engine

log = logging.getLogger("materialize")


def materialize(engine: Engine, store: OnlineStore, dataset: str = "test",
                units: list[int] | None = None) -> dict:
    t0 = time.perf_counter()
    where = "dataset = :d" + (" AND unit_id = ANY(:u)" if units else "")
    params = {"d": dataset, **({"u": list(units)} if units else {})}
    cols = ", ".join(["unit_id", "cycle", *FEATURE_SENSORS])
    hist = pd.read_sql(text(f"SELECT {cols} FROM cleaned.sensor_readings WHERE {where} ORDER BY unit_id, cycle"),
                       engine, params=params)
    latest = pd.read_sql(text(f"""
        SELECT DISTINCT ON (unit_id) unit_id, {", ".join(FEATURE_COLS)}
        FROM features.engine_features WHERE {where} ORDER BY unit_id, cycle DESC"""), engine, params=params)
    if hist.empty:
        raise ValueError(f"Нет данных для набора {dataset!r}")

    latest = latest.set_index("unit_id")
    for unit, g in hist.groupby("unit_id"):
        rows = [{"cycle": int(r.cycle), **{s: float(getattr(r, s)) for s in FEATURE_SENSORS}}
                for r in g.itertuples(index=False)]
        store.replace_history(int(unit), rows)
        f = latest.loc[unit]
        store.put_features(int(unit), {
            "cycle": int(f["cycle"]), "features": {c: float(f[c]) for c in FEATURE_COLS},
            "feature_set_version": FEATURE_SET_VERSION, "source": f"offline:{dataset}"})
    return {"dataset": dataset, "engines": int(hist["unit_id"].nunique()), "rows_read": int(len(hist)),
            "seconds": round(time.perf_counter() - t0, 2), "store": store.name}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="test", choices=["train", "test"])
    ap.add_argument("--store", default=os.getenv("PDM_ONLINE_STORE", "redis"))
    ap.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    a = ap.parse_args()
    res = materialize(get_engine(), make_online_store(a.store, a.redis_url), a.dataset)
    log.info("Материализовано: %s", res)
