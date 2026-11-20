"""Хранилище истории показаний двигателей для расчёта признаков в онлайн-режиме.

Хранится не вся история, а только то, что нужно признакам: первые BASELINE_CYCLES циклов («норма»)
и последние SLOPE_WINDOW циклов (окна среднего и тренда). Сейчас — в памяти процесса;
в блоке 4.1 добавится вариант на Redis с тем же интерфейсом.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Protocol

from src.data.features import BASELINE_CYCLES, SLOPE_WINDOW


def needed_rows(rows: list[dict]) -> list[dict]:
    """Оставляет только строки, нужные признакам: циклы «нормы» и последние SLOPE_WINDOW циклов."""
    tail = rows[-SLOPE_WINDOW:]
    first_tail_cycle = tail[0]["cycle"] if tail else 0
    return [r for r in rows if r["cycle"] <= BASELINE_CYCLES and r["cycle"] < first_tail_cycle] + tail


class HistoryStore(Protocol):
    def last_cycle(self, unit_id: int) -> int | None: ...
    def history(self, unit_id: int) -> list[dict]: ...
    def append(self, unit_id: int, reading: dict) -> None: ...
    def reset(self, unit_id: int) -> bool: ...
    def __len__(self) -> int: ...


class InMemoryHistoryStore:
    def __init__(self):
        self._base: dict[int, list[dict]] = {}
        self._recent: dict[int, deque] = {}
        self.lock = threading.RLock()   # запросы обрабатываются в нескольких потоках

    def last_cycle(self, unit_id: int) -> int | None:
        recent = self._recent.get(unit_id)
        return recent[-1]["cycle"] if recent else None

    def history(self, unit_id: int) -> list[dict]:
        base = self._base.get(unit_id, [])
        recent = [r for r in self._recent.get(unit_id, []) if r["cycle"] > BASELINE_CYCLES]
        return base + recent

    def append(self, unit_id: int, reading: dict) -> None:
        if reading["cycle"] <= BASELINE_CYCLES:
            self._base.setdefault(unit_id, []).append(reading)
        self._recent.setdefault(unit_id, deque(maxlen=SLOPE_WINDOW)).append(reading)

    def reset(self, unit_id: int) -> bool:
        existed = unit_id in self._recent
        self._base.pop(unit_id, None)
        self._recent.pop(unit_id, None)
        return existed

    def __len__(self) -> int:
        return len(self._recent)
