"""Тесты потоковой обработки без брокера Kafka: producer и consumer с подменёнными клиентами.

Главное, что проверяется: смещение фиксируется только после успешной записи в БД; при сбое записи
пачка перечитывается; плохие сообщения отбраковываются, не останавливая поток.
"""
import json
from collections import namedtuple

import pytest

from api.schemas import SensorReading
from src.data.load_raw import RAW_DIR, read_readings
from src.feature_store.online import InMemoryOnlineStore
from src.models.predict import FailurePredictor
from src.serving.service import PredictionService
from streaming.consumer import StreamProcessor, Stats, parse_record, run
from streaming.producer import bad_messages, iter_messages, parse_units, send_all

Record = namedtuple("Record", "topic partition offset key value")
TP = namedtuple("TP", "topic partition")


@pytest.fixture(scope="module")
def readings():
    return read_readings(RAW_DIR / "test_FD001.txt")


@pytest.fixture(scope="module")
def predictor():
    return FailurePredictor("final_v2")


def to_records(messages, partition=0, start=0):
    return [Record("sensor-readings", partition, start + i, str(k).encode(),
                   v if isinstance(v, bytes) else json.dumps(v).encode())
            for i, (k, v) in enumerate(messages)]


class FakeConsumer:
    """Отдаёт пачки по порядку; seek возвращает к нужному смещению; commit запоминает позиции."""

    def __init__(self, records, batch_size=50):
        self.records, self.batch_size = records, batch_size
        self.position = 0
        self.commits, self.seeks, self.events = [], [], []

    def poll(self, timeout_ms=0, max_records=500):
        batch = self.records[self.position:self.position + self.batch_size]
        self.position += len(batch)
        groups = {}
        for r in batch:
            groups.setdefault(TP(r.topic, r.partition), []).append(r)
        return groups

    def commit(self):
        self.commits.append(self.position)
        self.events.append(("commit", self.position))

    def seek(self, tp, offset):
        self.seeks.append(offset)
        self.position = next(i for i, r in enumerate(self.records) if r.offset == offset)


class MemoryWriter:
    """Имитация БД: дедупликация по (раздел, смещение), можно заставить падать."""

    def __init__(self, consumer=None, fail_times=0):
        self.raw, self.rejects, self.predictions = {}, [], []
        self.fail_times, self.consumer = fail_times, consumer

    def write_readings(self, readings, rejects):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("база недоступна")
        before = len(self.raw)
        for r in readings:
            self.raw.setdefault((r["source"], r["kafka_partition"], r["kafka_offset"]), r)
        self.rejects += rejects
        if self.consumer:
            self.consumer.events.append(("db_write", len(self.raw)))
        return len(self.raw) - before

    def write_predictions(self, predictions):
        self.predictions += predictions


# ---------- producer ----------

def test_parse_units():
    assert parse_units("1-3,7") == [1, 2, 3, 7] and parse_units(None) is None


def test_messages_are_valid_and_ordered_by_cycle_then_engine(readings):
    msgs = list(iter_messages(readings, unit_offset=1000, units=[1, 2], max_cycles=3))
    assert [(k, v["cycle"]) for k, v in msgs] == [(1001, 1), (1002, 1), (1001, 2), (1002, 2), (1001, 3), (1002, 3)]
    for _, body in msgs:
        SensorReading(**body)   # тело совпадает со схемой /predict


class FakeFuture:
    """Как Future из kafka-python: результат приходит в обработчики при flush()."""

    def __init__(self):
        self.callbacks, self.errbacks = [], []

    def add_callback(self, fn):
        self.callbacks.append(fn)

    def add_errback(self, fn):
        self.errbacks.append(fn)


class FakeProducer:
    """Раздел — по ключу (как разделитель Kafka); fail_keys — ключи, отправка которых завершится ошибкой."""

    def __init__(self, partitions=3, fail_keys=()):
        self.sent, self.flushed, self.pending = [], False, []
        self.partitions, self.fail_keys = partitions, set(fail_keys)

    def send(self, topic, key, value, headers):
        self.sent.append((topic, key, json.loads(value), dict(headers)))
        f = FakeFuture()
        self.pending.append((key, f))
        return f   # send() не ждёт брокера

    def flush(self):
        Meta = namedtuple("Meta", "partition")
        for key, f in self.pending:
            if key in self.fail_keys:
                [cb(TimeoutError("broker unavailable")) for cb in f.errbacks]
            else:
                [cb(Meta(int(key) % self.partitions)) for cb in f.callbacks]
        self.flushed = True


def test_send_all_async_delivery_report_and_routing(readings):
    p = FakeProducer()
    stats = send_all(p, iter_messages(readings, 1000, [5, 6, 7], 4), topic="t")
    assert stats["sent"] == stats["delivered"] == 12 and stats["failed"] == 0 and p.flushed
    assert stats["partitions"] == {0: 4, 1: 4, 2: 4}         # каждый двигатель — в свой раздел
    assert {s[1] for s in p.sent} == {b"1005", b"1006", b"1007"} and p.sent[0][3]["schema_version"] == b"1"


def test_send_all_counts_failed_deliveries(readings):
    p = FakeProducer(fail_keys={b"1005"})
    stats = send_all(p, iter_messages(readings, 1000, [5, 6], 3), topic="t")
    assert (stats["delivered"], stats["failed"]) == (3, 3) and "broker unavailable" in stats["errors"][0]


# ---------- разбор сообщений ----------

def test_parse_record_valid_and_rejects(readings):
    good = to_records(iter_messages(readings, 0, [1], 1))[0]
    reading, reject = parse_record(good)
    assert reject is None and reading["unit_id"] == 1 and reading["kafka_offset"] == 0
    bad = to_records(bad_messages(3, 0))
    errors = [parse_record(r)[1]["error"] for r in bad]
    assert errors[0].startswith("битый JSON")
    assert "s_2" in errors[1] and errors[1].startswith("валидация")   # нет обязательных датчиков
    assert "s_2" in errors[2]                                         # отрицательное показание


# ---------- consumer ----------

def test_commit_only_after_successful_db_write(readings):
    records = to_records(iter_messages(readings, 1000, [1, 2], 30))
    consumer = FakeConsumer(records, batch_size=25)
    writer = MemoryWriter(consumer)
    stats = run(consumer, StreamProcessor(writer, None), max_idle_seconds=0, poll_ms=0)
    assert stats.messages == 60 and stats.raw_inserted == 60 and stats.commits == 3
    # каждой фиксации смещения предшествует запись в БД
    for i, (kind, _) in enumerate(consumer.events):
        if kind == "commit":
            assert consumer.events[i - 1][0] == "db_write"


def test_db_failure_means_no_commit_and_batch_is_reread(readings):
    records = to_records(iter_messages(readings, 1000, [34], 40))   # у двигателя 34 более 200 циклов
    consumer = FakeConsumer(records, batch_size=20)
    writer = MemoryWriter(consumer, fail_times=2)          # первые две попытки записи падают
    stats = run(consumer, StreamProcessor(writer, None), max_idle_seconds=0, retry_delay=0, poll_ms=0)
    assert stats.db_failures == 2 and consumer.seeks == [0, 0]   # дважды вернулись к началу первой пачки
    assert stats.commits == 2 and consumer.commits == [20, 40]   # фиксация — только после успешной записи
    assert len(writer.raw) == 40 and stats.raw_inserted == 40    # ничего не потеряно и не задвоено


def test_redelivered_messages_are_not_duplicated_and_bad_ones_rejected(readings, predictor):
    records = to_records(list(iter_messages(readings, 1000, [3], 10)) + list(bad_messages(2, 1000)))
    redelivered = records[:5]                                    # повторная доставка (как после сбоя)
    consumer = FakeConsumer(records + redelivered, batch_size=100)
    writer = MemoryWriter()
    service = PredictionService(predictor, InMemoryOnlineStore())
    stats = run(consumer, StreamProcessor(writer, service), max_idle_seconds=0, poll_ms=0)
    assert stats.messages == 17 and stats.rejected == 2
    assert stats.raw_inserted == 10 and stats.duplicates_skipped == 5
    assert stats.predictions == 10 and stats.cycle_conflicts == 5    # повтор цикла не даёт второго прогноза
    assert [p["cycle"] for p in writer.predictions] == list(range(1, 11))
    assert all(p["source"] == "kafka" for p in writer.predictions)


def test_stream_predictions_equal_api_service(readings, predictor):
    """Прогноз по потоку совпадает с прогнозом того же сервиса, вызванного напрямую (как в API)."""
    msgs = list(iter_messages(readings, 2000, [34]))
    consumer = FakeConsumer(to_records(msgs), batch_size=64)
    writer = MemoryWriter()
    run(consumer, StreamProcessor(writer, PredictionService(predictor, InMemoryOnlineStore())),
        max_idle_seconds=0, poll_ms=0)
    direct = PredictionService(predictor, InMemoryOnlineStore())
    expected = [direct.process(body)["probability"] for _, body in msgs]
    assert [p["probability"] for p in writer.predictions] == pytest.approx(expected)
    assert writer.predictions[-1]["alarm"] is True   # двигатель 34 близок к отказу


def test_stats_summary_rate():
    s = Stats(messages=300, seconds=2.0)
    assert s.summary()["rate_per_s"] == 150.0
