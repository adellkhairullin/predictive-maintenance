"""Kafka producer — симулятор датчиков: проигрывает двигатели FD001 цикл за циклом в топик sensor-readings.

* Источник — PostgreSQL (raw.sensor_readings, по умолчанию) или исходные файлы (--source file).
* Маршрутизация: ключ сообщения — номер двигателя; разделитель Kafka (murmur2 по ключу) отправляет все
  сообщения двигателя в один раздел, поэтому они читаются по порядку.
* Сообщения разных двигателей перемешаны по времени, как поток от парка: сначала цикл 1 всех двигателей,
  затем цикл 2 и т. д.
* Асинхронная отправка: send() не ждёт брокера; подтверждения и ошибки приходят в обработчики
  (add_callback / add_errback), которые считают доставленные и неудачные сообщения по разделам.
* Повторы и надёжность: retries с паузой retry_backoff_ms, acks=all (запись подтверждена брокером),
  идемпотентный producer (повтор при сбое сети не создаёт дубль), flush в конце.
* Метаданные (время отправки, версия схемы) — в заголовках сообщения, тело совпадает со схемой /predict.

Запуск (нужны Kafka и PostgreSQL):
    python -m streaming.producer --units 1-5 --rate 50
    python -m streaming.producer --bad-messages 3      # добавить испорченные сообщения (проверка отбраковки)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd
from sqlalchemy import text

from src.data.load_raw import COLUMNS, RAW_DIR, read_readings
from streaming.config import BOOTSTRAP, SCHEMA_VERSION, TOPIC, ensure_topic

log = logging.getLogger("producer")


def load_readings(source: str, dataset: str, engine=None) -> pd.DataFrame:
    """Показания из PostgreSQL (raw.sensor_readings) или из исходного файла."""
    if source == "file":
        return read_readings(RAW_DIR / f"{dataset}_FD001.txt")
    if engine is None:
        from src.utils.db import get_engine
        engine = get_engine()
    df = pd.read_sql(text(f"SELECT {', '.join(COLUMNS)} FROM raw.sensor_readings WHERE dataset = :d "
                          "ORDER BY unit_id, cycle"), engine, params={"d": dataset})
    if df.empty:
        raise ValueError(f"В raw.sensor_readings нет данных набора {dataset!r} — сначала выполните ETL")
    return df


def parse_units(spec: str | None) -> list[int] | None:
    """'1-5,10' -> [1, 2, 3, 4, 5, 10]; None -> все двигатели."""
    if not spec:
        return None
    units = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        units.extend(range(int(a), int(b or a) + 1))
    return units


def iter_messages(df: pd.DataFrame, unit_offset: int = 0, units: list[int] | None = None,
                  max_cycles: int | None = None) -> Iterator[tuple[int, dict]]:
    """(ключ, тело) в порядке «по циклам, внутри цикла — по двигателям»."""
    d = df if units is None else df[df["unit_id"].isin(units)]
    if max_cycles:
        d = d[d["cycle"] <= max_cycles]
    for r in d.sort_values(["cycle", "unit_id"]).to_dict("records"):
        body = {k: (int(v) if k in ("unit_id", "cycle") else float(v)) for k, v in r.items()}
        body["unit_id"] += unit_offset
        yield body["unit_id"], body


def bad_messages(n: int, unit_offset: int) -> Iterator[tuple[int, bytes]]:
    """Испорченные сообщения для проверки отбраковки: битый JSON, нет датчика, отрицательное показание."""
    samples = [b"{not json", json.dumps({"unit_id": unit_offset + 999, "cycle": 1}).encode(),
               json.dumps({"unit_id": unit_offset + 999, "cycle": 1, "s_2": -1}).encode()]
    for i in range(n):
        yield unit_offset + 999, samples[i % len(samples)]


class DeliveryReport:
    """Обработчики асинхронной отправки: подтверждения брокера и ошибки, по разделам."""

    def __init__(self):
        self.delivered, self.failed, self.partitions, self.errors = 0, 0, {}, []

    def on_success(self, metadata) -> None:
        self.delivered += 1
        self.partitions[metadata.partition] = self.partitions.get(metadata.partition, 0) + 1

    def on_error(self, exc) -> None:
        self.failed += 1
        if len(self.errors) < 5:
            self.errors.append(repr(exc))
        log.error("Сообщение не доставлено после всех повторов: %r", exc)


def send_all(producer, messages, topic: str = TOPIC, rate: float = 0.0) -> dict:
    """Асинхронно отправляет сообщения; rate — сообщений в секунду (0 — без ограничения).

    send() сразу возвращает Future; результат приходит в обработчики DeliveryReport.
    flush() в конце ждёт подтверждения всех сообщений.
    """
    report = DeliveryReport()
    t0 = time.perf_counter()
    n = 0
    for key, value in messages:
        headers = [("produced_at", datetime.now(timezone.utc).isoformat().encode()),
                   ("schema_version", SCHEMA_VERSION.encode())]
        payload = value if isinstance(value, bytes) else json.dumps(value).encode()
        future = producer.send(topic, key=str(key).encode(), value=payload, headers=headers)
        future.add_callback(report.on_success)
        future.add_errback(report.on_error)
        n += 1
        if rate > 0:
            wait = t0 + n / rate - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
    producer.flush()
    dt = time.perf_counter() - t0
    return {"sent": n, "delivered": report.delivered, "failed": report.failed,
            "partitions": dict(sorted(report.partitions.items())), "errors": report.errors,
            "seconds": round(dt, 2), "rate_per_s": round(n / dt, 1) if dt else None}


def make_producer(bootstrap: str = BOOTSTRAP):
    from kafka import KafkaProducer
    return KafkaProducer(bootstrap_servers=bootstrap, acks="all", enable_idempotence=True,
                         retries=10, retry_backoff_ms=200, linger_ms=5,
                         client_id="pdm-sensor-simulator")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Симулятор датчиков: показания FD001 -> Kafka")
    ap.add_argument("--dataset", default="test", choices=["train", "test"])
    ap.add_argument("--source", default="db", choices=["db", "file"],
                    help="db — raw.sensor_readings в PostgreSQL (по умолчанию), file — исходный файл")
    ap.add_argument("--units", help="двигатели, например 1-5,10 (по умолчанию все)")
    ap.add_argument("--max-cycles", type=int, help="ограничить число циклов каждого двигателя")
    ap.add_argument("--unit-offset", type=int, default=1000,
                    help="прибавить к номеру двигателя (по умолчанию 1000: не пересекаться с двигателями 1–100, "
                         "материализованными в онлайн-хранилище)")
    ap.add_argument("--rate", type=float, default=0.0, help="сообщений в секунду, 0 — без ограничения")
    ap.add_argument("--bad-messages", type=int, default=0, help="добавить N испорченных сообщений")
    ap.add_argument("--bootstrap", default=BOOTSTRAP)
    ap.add_argument("--topic", default=TOPIC)
    a = ap.parse_args(argv)

    created = ensure_topic(a.bootstrap, a.topic)
    log.info("Топик %s %s", a.topic, "создан" if created else "уже существует")
    df = load_readings(a.source, a.dataset)
    messages = list(iter_messages(df, a.unit_offset, parse_units(a.units), a.max_cycles))
    messages += list(bad_messages(a.bad_messages, a.unit_offset))
    producer = make_producer(a.bootstrap)
    try:
        stats = send_all(producer, messages, a.topic, a.rate)
    finally:
        producer.close()
    stats.update({"source": a.source, "valid": len(messages) - a.bad_messages, "bad": a.bad_messages,
                  "engines": len({k for k, _ in messages}) - (1 if a.bad_messages else 0)})
    log.info("Отправлено: %s", stats)
    print(json.dumps(stats, ensure_ascii=False))
    if stats["failed"]:
        sys.exit(1)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    logging.getLogger("kafka").setLevel(logging.WARNING)
    main()
