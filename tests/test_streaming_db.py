"""Интеграционный тест записи потока в PostgreSQL (без Kafka): дедупликация и отбраковка.
Пропускается, если база недоступна."""
from collections import namedtuple

import pytest
from sqlalchemy import text

from streaming.consumer import DbWriter, StreamProcessor, Stats, parse_record

pytestmark = pytest.mark.integration
Record = namedtuple("Record", "topic partition offset key value")
TOPIC = "pytest-stream"
BODY = ('{"unit_id": 987001, "cycle": %d, "s_2": 641.8, "s_3": 1589.7, "s_4": 1400.6, "s_7": 554.4, '
        '"s_8": 2388.1, "s_9": 9046.2, "s_11": 47.5, "s_12": 521.7, "s_13": 2388.0, "s_14": 8138.6, '
        '"s_15": 8.42, "s_17": 392, "s_20": 39.1, "s_21": 23.42}')


@pytest.fixture
def writer():
    try:
        from src.utils.db import get_engine
        w = DbWriter(get_engine())
    except Exception as exc:
        pytest.skip(f"PostgreSQL недоступен: {exc}")
    yield w
    with w.engine.begin() as conn:
        conn.execute(text("DELETE FROM raw.sensor_readings WHERE source = :s"), {"s": f"kafka:{TOPIC}"})
        conn.execute(text("DELETE FROM raw.stream_rejects WHERE source = :s"), {"s": f"kafka:{TOPIC}"})


def test_redelivery_is_deduplicated_and_rejects_stored(writer):
    records = [Record(TOPIC, 0, i, b"987001", (BODY % (i + 1)).encode()) for i in range(3)]
    records.append(Record(TOPIC, 0, 3, b"987001", b"{broken"))
    proc = StreamProcessor(writer, service=None)
    s1, s2 = Stats(), Stats()
    proc.handle_batch(records, s1)
    proc.handle_batch(records, s2)          # та же пачка ещё раз — как повторная доставка после сбоя
    assert (s1.raw_inserted, s1.rejected) == (3, 1)
    assert (s2.raw_inserted, s2.duplicates_skipped) == (0, 3)
    with writer.engine.connect() as conn:
        n_raw = conn.execute(text("SELECT count(*) FROM raw.sensor_readings WHERE source = :s AND dataset = 'stream'"),
                             {"s": f"kafka:{TOPIC}"}).scalar_one()
        rej = conn.execute(text("SELECT kafka_offset, error FROM raw.stream_rejects WHERE source = :s"),
                           {"s": f"kafka:{TOPIC}"}).all()
    assert n_raw == 3
    assert len(rej) == 1 and rej[0].kafka_offset == 3 and rej[0].error.startswith("битый JSON")


def test_producer_reads_readings_from_db(writer):
    """Producer берёт показания из PostgreSQL; они совпадают с исходным файлом."""
    from src.data.load_raw import RAW_DIR, read_readings
    from streaming.producer import iter_messages, load_readings
    db = load_readings("db", "test", writer.engine)
    file = read_readings(RAW_DIR / "test_FD001.txt")
    assert len(db) == len(file) == 13096
    a = list(iter_messages(db, 1000, [34], 5))
    b = list(iter_messages(file, 1000, [34], 5))
    assert a == b


def test_parse_record_keeps_kafka_coordinates():
    reading, _ = parse_record(Record(TOPIC, 2, 41, b"1", (BODY % 1).encode()))
    assert (reading["source"], reading["kafka_partition"], reading["kafka_offset"]) == (f"kafka:{TOPIC}", 2, 41)
