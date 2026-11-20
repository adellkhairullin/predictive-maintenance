"""Журнал предсказаний в PostgreSQL (monitoring.predictions).

Запрос только кладёт запись в очередь в памяти (микросекунды), а отдельный поток раз в секунду
пишет накопившиеся записи в базу одной пачкой. Так журнал не увеличивает время ответа. Если база недоступна, сервис продолжает
работать без журнала.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Protocol

from sqlalchemy import text

log = logging.getLogger("api.prediction_log")

INSERT = text("""
    INSERT INTO monitoring.predictions
        (unit_id, cycle, model_version, probability, alarm, threshold, latency_ms, warnings, source)
    VALUES (:unit_id, :cycle, :model_version, :probability, :alarm, :threshold, :latency_ms, :warnings, :source)
""")
MAX_BATCH = 500


class PredictionLogger(Protocol):
    name: str
    def log(self, record: dict) -> None: ...
    def close(self) -> None: ...


class NullPredictionLogger:
    name = "none"

    def log(self, record: dict) -> None:
        pass

    def close(self) -> None:
        pass


class DbPredictionLogger:
    name = "db"

    def __init__(self, engine, flush_interval: float = 1.0):
        from src.utils.db import SQL_DIR, run_sql_file
        self.engine = engine
        self.flush_interval = flush_interval
        run_sql_file(engine, SQL_DIR / "schema.sql")   # создаёт monitoring.predictions, если её нет
        self._queue: queue.Queue = queue.Queue(maxsize=100_000)
        self._worker = threading.Thread(target=self._run, name="prediction-log", daemon=True)
        self._worker.start()

    def log(self, record: dict) -> None:
        try:
            self._queue.put_nowait({"source": "api", **record})
        except queue.Full:
            log.warning("Очередь журнала переполнена, запись пропущена")

    def _run(self) -> None:
        """Копит записи не дольше flush_interval секунд и пишет их одной пачкой.

        Запись по одной строке после каждого запроса мешала потоку, который отправляет ответы
        (в Python в каждый момент выполняется один поток), и увеличивала время ответа.
        """
        stop = False
        while not stop:
            first = self._queue.get()          # ждём первую запись
            got = 1
            batch, stop = ([], True) if first is None else ([first], False)
            deadline = time.monotonic() + self.flush_interval
            while not stop and len(batch) < MAX_BATCH:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    nxt = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                got += 1
                if nxt is None:                # сигнал остановки: дописать пачку и выйти
                    stop = True
                else:
                    batch.append(nxt)
            if batch:
                self._write(batch)
            for _ in range(got):
                self._queue.task_done()

    def _write(self, batch: list[dict]) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(INSERT, batch)
            return
        except Exception:
            if len(batch) == 1:
                log.exception("Не удалось записать предсказание в журнал: %s", batch[0])
                return
        for rec in batch:   # пачка не записалась — пишем по одной, чтобы потерять только плохие записи
            self._write([rec])

    def flush(self) -> None:
        """Дождаться, пока все записи из очереди попадут в базу."""
        self._queue.join()

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=10)


def make_logger(mode: str) -> PredictionLogger:
    if mode == "db":
        try:
            from src.utils.db import get_engine
            return DbPredictionLogger(get_engine())
        except Exception as exc:
            log.warning("Журнал предсказаний отключён: база недоступна (%s)", exc)
    return NullPredictionLogger()
