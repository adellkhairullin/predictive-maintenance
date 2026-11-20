"""Онлайн-хранилище признаков: история двигателей и последний вектор признаков.

Для расчёта признаков в онлайн-режиме нужна не вся история двигателя, а первые BASELINE_CYCLES циклов
(«норма») и последние SLOPE_WINDOW циклов (окна среднего и тренда). Их и хранит онлайн-хранилище,
вместе с последним рассчитанным вектором признаков и предсказанием.

Два варианта с одним интерфейсом:
* InMemoryOnlineStore — в памяти процесса (разработка, тесты);
* RedisOnlineStore — в Redis: переживает перезапуск сервиса и общий для нескольких его копий.

Ключи Redis (prefix по умолчанию "pdm"):
    pdm:engine:{id}:base      список JSON — циклы 1..BASELINE_CYCLES
    pdm:engine:{id}:recent    список JSON — последние SLOPE_WINDOW циклов
    pdm:engine:{id}:last      номер последнего принятого цикла
    pdm:engine:{id}:features  JSON — последний вектор признаков и предсказание
    pdm:engines               множество номеров двигателей
"""
from __future__ import annotations

import json
import threading
from collections import deque
from typing import Protocol

from src.data.features import BASELINE_CYCLES, SLOPE_WINDOW


class CycleConflict(Exception):
    """Цикл подан не по порядку (пропуск или повтор) или его уже принял другой запрос."""

    def __init__(self, unit_id: int, received: int, expected: int):
        self.unit_id, self.received, self.expected = unit_id, received, expected
        kind = "повтор уже обработанного цикла" if received < expected else "пропуск циклов"
        super().__init__(f"Цикл подан не по порядку: {kind}")


def needed_rows(rows: list[dict]) -> list[dict]:
    """Оставляет только строки, нужные признакам: циклы «нормы» и последние SLOPE_WINDOW циклов."""
    tail = rows[-SLOPE_WINDOW:]
    first_tail_cycle = tail[0]["cycle"] if tail else 0
    return [r for r in rows if r["cycle"] <= BASELINE_CYCLES and r["cycle"] < first_tail_cycle] + tail


class OnlineStore(Protocol):
    name: str
    def history(self, unit_id: int) -> list[dict]: ...
    def last_cycle(self, unit_id: int) -> int | None: ...
    def append(self, unit_id: int, row: dict, expected_last: int | None) -> None: ...
    def replace_history(self, unit_id: int, rows: list[dict]) -> None: ...
    def put_features(self, unit_id: int, record: dict) -> None: ...
    def get_features(self, unit_id: int) -> dict | None: ...
    def reset(self, unit_id: int) -> bool: ...
    def count(self) -> int: ...


def _expected(last: int | None) -> int:
    return 1 if last is None else last + 1


class InMemoryOnlineStore:
    name = "memory"

    def __init__(self):
        self._base: dict[int, list[dict]] = {}
        self._recent: dict[int, deque] = {}
        self._features: dict[int, dict] = {}
        self._lock = threading.RLock()   # запросы обрабатываются в нескольких потоках

    def history(self, unit_id: int) -> list[dict]:
        with self._lock:
            base = list(self._base.get(unit_id, []))
            recent = [r for r in self._recent.get(unit_id, []) if r["cycle"] > BASELINE_CYCLES]
        return base + recent

    def last_cycle(self, unit_id: int) -> int | None:
        recent = self._recent.get(unit_id)
        return recent[-1]["cycle"] if recent else None

    def append(self, unit_id: int, row: dict, expected_last: int | None) -> None:
        with self._lock:   # проверка и добавление — атомарно
            last = self.last_cycle(unit_id)
            if last != expected_last or row["cycle"] != _expected(last):
                raise CycleConflict(unit_id, row["cycle"], _expected(last))
            if row["cycle"] <= BASELINE_CYCLES:
                self._base.setdefault(unit_id, []).append(row)
            self._recent.setdefault(unit_id, deque(maxlen=SLOPE_WINDOW)).append(row)

    def replace_history(self, unit_id: int, rows: list[dict]) -> None:
        rows = needed_rows(sorted(rows, key=lambda r: r["cycle"]))
        with self._lock:
            self._base[unit_id] = [r for r in rows if r["cycle"] <= BASELINE_CYCLES]
            self._recent[unit_id] = deque(rows[-SLOPE_WINDOW:], maxlen=SLOPE_WINDOW)

    def put_features(self, unit_id: int, record: dict) -> None:
        self._features[unit_id] = record

    def get_features(self, unit_id: int) -> dict | None:
        return self._features.get(unit_id)

    def reset(self, unit_id: int) -> bool:
        with self._lock:
            existed = unit_id in self._recent
            for d in (self._base, self._recent, self._features):
                d.pop(unit_id, None)
        return existed

    def count(self) -> int:
        return len(self._recent)


class RedisOnlineStore:
    name = "redis"

    def __init__(self, client, prefix: str = "pdm", max_retries: int = 5):
        self.r = client          # redis.Redis(decode_responses=True)
        self.prefix = prefix
        self.max_retries = max_retries

    def _key(self, unit_id: int, part: str) -> str:
        return f"{self.prefix}:engine:{unit_id}:{part}"

    @property
    def _engines_key(self) -> str:
        return f"{self.prefix}:engines"

    def history(self, unit_id: int) -> list[dict]:
        pipe = self.r.pipeline(transaction=False)
        pipe.lrange(self._key(unit_id, "base"), 0, -1)
        pipe.lrange(self._key(unit_id, "recent"), 0, -1)
        base, recent = pipe.execute()
        base = [json.loads(x) for x in base]
        recent = [r for r in (json.loads(x) for x in recent) if r["cycle"] > BASELINE_CYCLES]
        return base + recent

    def last_cycle(self, unit_id: int) -> int | None:
        v = self.r.get(self._key(unit_id, "last"))
        return int(v) if v is not None else None

    def append(self, unit_id: int, row: dict, expected_last: int | None) -> None:
        """Атомарно: если последний цикл всё ещё expected_last — добавить row, иначе CycleConflict.

        WATCH отменяет транзакцию, если ключ last изменил другой запрос (другая копия сервиса)
        между проверкой и записью; тогда проверка повторяется.
        """
        from redis.exceptions import WatchError

        last_key = self._key(unit_id, "last")
        payload = json.dumps(row)
        for _ in range(self.max_retries):
            with self.r.pipeline() as pipe:
                try:
                    pipe.watch(last_key)
                    v = pipe.get(last_key)
                    last = int(v) if v is not None else None
                    if last != expected_last or row["cycle"] != _expected(last):
                        pipe.unwatch()
                        raise CycleConflict(unit_id, row["cycle"], _expected(last))
                    pipe.multi()
                    if row["cycle"] <= BASELINE_CYCLES:
                        pipe.rpush(self._key(unit_id, "base"), payload)
                    pipe.rpush(self._key(unit_id, "recent"), payload)
                    pipe.ltrim(self._key(unit_id, "recent"), -SLOPE_WINDOW, -1)
                    pipe.set(last_key, row["cycle"])
                    pipe.sadd(self._engines_key, unit_id)
                    pipe.execute()
                    return
                except WatchError:
                    continue
        last = self.last_cycle(unit_id)
        raise CycleConflict(unit_id, row["cycle"], _expected(last))

    def replace_history(self, unit_id: int, rows: list[dict]) -> None:
        rows = needed_rows(sorted(rows, key=lambda r: r["cycle"]))
        base = [json.dumps(r) for r in rows if r["cycle"] <= BASELINE_CYCLES]
        recent = [json.dumps(r) for r in rows[-SLOPE_WINDOW:]]
        pipe = self.r.pipeline()   # MULTI/EXEC: история заменяется целиком
        pipe.delete(self._key(unit_id, "base"), self._key(unit_id, "recent"), self._key(unit_id, "features"))
        if base:
            pipe.rpush(self._key(unit_id, "base"), *base)
        if recent:
            pipe.rpush(self._key(unit_id, "recent"), *recent)
            pipe.set(self._key(unit_id, "last"), rows[-1]["cycle"])
            pipe.sadd(self._engines_key, unit_id)
        pipe.execute()

    def put_features(self, unit_id: int, record: dict) -> None:
        self.r.set(self._key(unit_id, "features"), json.dumps(record))

    def get_features(self, unit_id: int) -> dict | None:
        v = self.r.get(self._key(unit_id, "features"))
        return json.loads(v) if v is not None else None

    def reset(self, unit_id: int) -> bool:
        pipe = self.r.pipeline()
        pipe.delete(*(self._key(unit_id, p) for p in ("base", "recent", "last", "features")))
        pipe.srem(self._engines_key, unit_id)
        deleted, _ = pipe.execute()
        return deleted > 0

    def count(self) -> int:
        return int(self.r.scard(self._engines_key))


def make_online_store(kind: str = "memory", redis_url: str | None = None) -> OnlineStore:
    """memory — в памяти; redis — в Redis по адресу redis_url (при недоступности — ошибка, без тихой подмены)."""
    if kind == "memory":
        return InMemoryOnlineStore()
    if kind == "redis":
        import redis
        client = redis.Redis.from_url(redis_url or "redis://localhost:6379/0", decode_responses=True)
        client.ping()
        return RedisOnlineStore(client)
    raise ValueError(f"Неизвестный тип онлайн-хранилища: {kind}")
