"""Kafka consumer: показания датчиков -> PostgreSQL (raw) -> признаки и прогноз -> журнал предсказаний.

Порядок обработки пачки сообщений:
1. Проверка каждого сообщения схемой /predict (api.schemas.SensorReading). Не прошедшие проверку
   (битый JSON, ошибки валидации) — в raw.stream_rejects; поток из-за них не останавливается.
2. Запись в PostgreSQL одной транзакцией: показания в raw.sensor_readings (dataset='stream') и отбраковка.
   Повторно доставленное сообщение не записывается дважды: уникальный индекс (раздел, смещение)
   и INSERT ... ON CONFLICT DO NOTHING.
3. Онлайн-обработка (тот же сервис, что в API): признаки по истории в онлайн-хранилище, прогноз;
   прогнозы — в monitoring.predictions (source='kafka', не больше одного на цикл и версию модели).
   Онлайн-хранилище — кэш, источник истины — raw в PostgreSQL. Если порядок циклов нарушен
   (повторная доставка после сбоя или перераспределения разделов между экземплярами):
   * цикл уже обработан -> прогноз пересчитывается по истории из raw (дубль прогноза не запишется);
   * в истории Redis «дыра» -> история двигателя восстанавливается из raw, цикл обрабатывается.
4. Только после успешной записи — фиксация смещения в Kafka (commit). Автофиксация отключена.
   Если запись не удалась, смещение не фиксируется, consumer возвращается к началу пачки (seek)
   и повторяет попытку; при падении процесса пачка будет прочитана заново после перезапуска.
   Если фиксация отклонена из-за перераспределения разделов, сообщения получит новый владелец раздела.
Итог: доставка «хотя бы один раз», а в raw и в журнале прогнозов — без дублей и без пропусков.

Запуск (нужны Kafka, PostgreSQL и онлайн-хранилище):
    python -m streaming.consumer                       # работает постоянно
    python -m streaming.consumer --max-idle-seconds 20 # завершиться, если 20 с нет сообщений
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.schemas import SensorReading
from src.data.features import FEATURE_SENSORS
from src.data.load_raw import SENSOR_COLS, SETTING_COLS
from src.feature_store.online import CycleConflict, make_online_store
from src.models.input_ranges import load_ranges
from src.models.predict import FailurePredictor
from src.serving.service import PredictionService
from src.utils.db import SQL_DIR, get_engine, run_sql_file
from streaming.config import BOOTSTRAP, GROUP_ID, TOPIC, ensure_topic

log = logging.getLogger("consumer")
READING_COLS = ["unit_id", "cycle", *SETTING_COLS, *SENSOR_COLS]

INSERT_RAW = text(f"""
    INSERT INTO raw.sensor_readings (dataset, {", ".join(READING_COLS)}, source, kafka_partition, kafka_offset)
    VALUES ('stream', {", ".join(":" + c for c in READING_COLS)}, :source, :kafka_partition, :kafka_offset)
    ON CONFLICT (source, kafka_partition, kafka_offset) WHERE kafka_offset IS NOT NULL DO NOTHING
""")
INSERT_REJECT = text("""
    INSERT INTO raw.stream_rejects (source, kafka_partition, kafka_offset, payload, error)
    VALUES (:source, :kafka_partition, :kafka_offset, :payload, :error)
    ON CONFLICT (source, kafka_partition, kafka_offset) DO NOTHING
""")
INSERT_PREDICTION = text("""
    INSERT INTO monitoring.predictions
        (unit_id, cycle, model_version, probability, alarm, threshold, latency_ms, warnings, source)
    VALUES (:unit_id, :cycle, :model_version, :probability, :alarm, :threshold, :latency_ms, :warnings, 'kafka')
    ON CONFLICT (unit_id, cycle, model_version) WHERE source = 'kafka' DO NOTHING
""")
RAW_HISTORY = text(f"""
    SELECT DISTINCT ON (cycle) cycle, {", ".join(FEATURE_SENSORS)}
    FROM raw.sensor_readings
    WHERE dataset = 'stream' AND unit_id = :u AND cycle <= :c
    ORDER BY cycle, id
""")
REBALANCE_ERRORS = {"CommitFailedError", "RebalanceInProgressError", "IllegalGenerationError",
                    "UnknownMemberIdError"}


@dataclass
class Stats:
    batches: int = 0
    messages: int = 0
    raw_inserted: int = 0
    duplicates_skipped: int = 0
    rejected: int = 0
    predictions: int = 0
    alarms: int = 0
    cycle_conflicts: int = 0
    redelivered_recomputed: int = 0
    gaps_repaired: int = 0
    unrecoverable: int = 0
    db_failures: int = 0
    commit_rejected: int = 0
    commits: int = 0
    seconds: float = 0.0
    errors: list = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d["errors"] = d["errors"][-5:]
        d["rate_per_s"] = round(self.messages / self.seconds, 1) if self.seconds else None
        return d


def parse_record(rec) -> tuple[dict | None, dict | None]:
    """(показание, None) или (None, отбраковка)."""
    meta = {"source": f"kafka:{rec.topic}", "kafka_partition": rec.partition, "kafka_offset": rec.offset}
    raw = rec.value.decode("utf-8", errors="replace") if isinstance(rec.value, bytes) else str(rec.value)
    try:
        reading = SensorReading(**json.loads(raw)).model_dump()
    except (json.JSONDecodeError, TypeError) as exc:
        return None, {**meta, "payload": raw[:2000], "error": f"битый JSON: {exc}"}
    except ValidationError as exc:
        err = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return None, {**meta, "payload": raw[:2000], "error": f"валидация: {err}"[:1000]}
    return {**reading, **meta}, None


class DbWriter:
    """Запись в PostgreSQL. Показания и отбраковка — одной транзакцией."""

    def __init__(self, engine: Engine):
        self.engine = engine
        run_sql_file(engine, SQL_DIR / "schema.sql")   # колонки Kafka, индекс, таблица отбраковки

    def write_readings(self, readings: list[dict], rejects: list[dict]) -> int:
        inserted = 0
        with self.engine.begin() as conn:
            for r in readings:   # построчно, чтобы знать, сколько строк реально вставлено (без дублей)
                inserted += conn.execute(INSERT_RAW, {c: r.get(c) for c in
                                                      READING_COLS + ["source", "kafka_partition", "kafka_offset"]}
                                         ).rowcount
            if rejects:
                conn.execute(INSERT_REJECT, rejects)
        return inserted

    def write_predictions(self, predictions: list[dict]) -> int:
        """Возвращает число реально записанных прогнозов (повторы по циклу отбрасываются)."""
        inserted = 0
        if predictions:
            with self.engine.begin() as conn:
                for p in predictions:
                    inserted += conn.execute(INSERT_PREDICTION, p).rowcount
        return inserted

    def raw_history(self, unit_id: int, upto_cycle: int) -> list[dict]:
        """История двигателя из потока (raw, dataset='stream'), циклы 1..upto_cycle, по одной строке на цикл."""
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(RAW_HISTORY, {"u": unit_id, "c": upto_cycle})]


PREDICTION_KEYS = ("unit_id", "cycle", "model_version", "probability", "alarm", "threshold", "latency_ms", "warnings")


class StreamProcessor:
    def __init__(self, writer, service: PredictionService | None):
        self.writer, self.service = writer, service

    def _recover(self, r: dict, exc: CycleConflict, stats: Stats) -> dict | None:
        """Порядок циклов нарушен: восстановить по истории из raw (источник истины)."""
        stats.cycle_conflicts += 1
        rows = self.writer.raw_history(r["unit_id"], r["cycle"])
        if [row["cycle"] for row in rows] != list(range(1, r["cycle"] + 1)):
            stats.unrecoverable += 1   # в raw нет полной истории — прогноз по неполным данным не строим
            stats.errors.append(f"{r['unit_id']}/{r['cycle']}: {exc}; в raw неполная история")
            return None
        if exc.received < exc.expected:          # цикл уже обработан: повторная доставка
            stats.redelivered_recomputed += 1
            return {"unit_id": r["unit_id"], **self.service.predict_history(rows)}
        stats.gaps_repaired += 1                 # «дыра» в истории онлайн-хранилища
        self.service.store.replace_history(r["unit_id"], rows[:-1])
        try:
            return self.service.process(r)
        except CycleConflict:                    # цикл успел обработать другой экземпляр
            stats.redelivered_recomputed += 1
            return {"unit_id": r["unit_id"], **self.service.predict_history(rows)}

    def handle_batch(self, records, stats: Stats) -> None:
        readings, rejects = [], []
        for rec in records:
            reading, reject = parse_record(rec)
            (readings if reading else rejects).append(reading or reject)

        # 1) запись в БД (если упадёт — исключение, смещение не будет зафиксировано)
        inserted = self.writer.write_readings(readings, rejects)

        # 2) онлайн-обработка: признаки и прогноз
        predictions, written = [], 0
        if self.service is not None:
            for r in sorted(readings, key=lambda x: (x["unit_id"], x["cycle"])):
                try:
                    res = self.service.process(r)
                except CycleConflict as exc:
                    res = self._recover(r, exc, stats)
                    if res is None:
                        continue
                predictions.append({k: res[k] for k in PREDICTION_KEYS})
            written = self.writer.write_predictions(predictions)

        stats.batches += 1
        stats.messages += len(records)
        stats.raw_inserted += inserted
        stats.duplicates_skipped += len(readings) - inserted
        stats.rejected += len(rejects)
        stats.predictions += written
        stats.alarms += sum(p["alarm"] for p in predictions)


def run(consumer, processor: StreamProcessor, max_idle_seconds: float | None = None,
        max_messages: int | None = None, retry_delay: float = 2.0, poll_ms: int = 1000) -> Stats:
    """Цикл чтения. Смещение фиксируется только после успешной обработки пачки."""
    stats = Stats()
    t0 = None                       # время первого сообщения: от него считается скорость обработки
    last_data = time.monotonic()
    while True:
        batch = consumer.poll(timeout_ms=poll_ms, max_records=500)
        records = [r for recs in batch.values() for r in recs]
        if not records:
            if max_idle_seconds is not None and time.monotonic() - last_data >= max_idle_seconds:
                break
            continue
        t0 = t0 or time.monotonic()
        try:
            processor.handle_batch(records, stats)
        except Exception as exc:
            stats.db_failures += 1
            stats.errors.append(f"пачка не обработана: {exc}")
            log.exception("Пачка не обработана — смещение НЕ фиксируется, повтор через %s с", retry_delay)
            for tp, recs in batch.items():
                consumer.seek(tp, recs[0].offset)   # перечитать ту же пачку
            time.sleep(retry_delay)
            continue
        try:
            consumer.commit()   # только после успешной записи в БД
        except Exception as exc:
            if type(exc).__name__ not in REBALANCE_ERRORS:
                raise
            # разделы перешли к другому экземпляру группы: он прочитает пачку заново,
            # а дубли отсеются уникальными индексами raw и журнала прогнозов
            stats.commit_rejected += 1
            log.warning("Фиксация смещения отклонена (перераспределение разделов): %r", exc)
            continue
        stats.commits += 1
        last_data = time.monotonic()
        stats.seconds = round(last_data - t0, 2)
        log.info("пачка %s: сообщений %s, в raw %s, отбраковано %s, прогнозов %s, тревог %s, "
                 "конфликтов порядка %s (пересчитано %s, восстановлено дыр %s, без истории %s)",
                 stats.batches, len(records), stats.raw_inserted, stats.rejected, stats.predictions, stats.alarms,
                 stats.cycle_conflicts, stats.redelivered_recomputed, stats.gaps_repaired, stats.unrecoverable)
        if max_messages and stats.messages >= max_messages:
            break
    return stats


def make_consumer(bootstrap: str = BOOTSTRAP, topic: str = TOPIC, group_id: str = GROUP_ID):
    from kafka import KafkaConsumer
    return KafkaConsumer(topic, bootstrap_servers=bootstrap, group_id=group_id,
                         enable_auto_commit=False, auto_offset_reset="earliest",
                         client_id="pdm-consumer", max_poll_records=500)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Kafka -> PostgreSQL -> прогноз")
    ap.add_argument("--bootstrap", default=BOOTSTRAP)
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--group-id", default=GROUP_ID)
    ap.add_argument("--max-idle-seconds", type=float, help="завершиться, если столько секунд нет сообщений")
    ap.add_argument("--no-predict", action="store_true", help="только запись в raw, без прогноза")
    a = ap.parse_args(argv)

    service = None
    if not a.no_predict:
        store = make_online_store(os.getenv("PDM_ONLINE_STORE", "redis"), os.getenv("REDIS_URL"))
        service = PredictionService(FailurePredictor(os.getenv("PDM_MODEL_NAME", "final_v2")), store, load_ranges())
        service.warm_up()
    ensure_topic(a.bootstrap, a.topic)
    consumer = make_consumer(a.bootstrap, a.topic, a.group_id)
    try:
        stats = run(consumer, StreamProcessor(DbWriter(get_engine()), service), a.max_idle_seconds)
    finally:
        consumer.close()
    log.info("Итог: %s", stats.summary())
    print(json.dumps(stats.summary(), ensure_ascii=False))
    return stats.summary()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    logging.getLogger("kafka").setLevel(logging.WARNING)
    main()
