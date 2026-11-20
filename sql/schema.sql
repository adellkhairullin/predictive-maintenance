-- Схемы слоёв данных и таблицы слоя raw.
-- Скрипт можно запускать повторно: объекты создаются, только если их ещё нет.

CREATE SCHEMA IF NOT EXISTS raw;       -- данные как есть из источника
CREATE SCHEMA IF NOT EXISTS cleaned;   -- после проверок и очистки
CREATE SCHEMA IF NOT EXISTS enriched;  -- с расчётным остаточным ресурсом и целевой переменной
CREATE SCHEMA IF NOT EXISTS features;  -- признаки для модели
CREATE SCHEMA IF NOT EXISTS monitoring; -- журнал предсказаний сервиса

-- Журнал предсказаний API: для контроля качества модели и задержки (см. блок 1.3).
CREATE TABLE IF NOT EXISTS monitoring.predictions (
    id            bigserial PRIMARY KEY,
    unit_id       integer          NOT NULL,
    cycle         integer          NOT NULL,
    model_version varchar(32)      NOT NULL,
    probability   double precision NOT NULL,
    alarm         boolean          NOT NULL,
    threshold     double precision NOT NULL,
    latency_ms    double precision NOT NULL,
    warnings      text[]           NOT NULL DEFAULT '{}',
    source        varchar(32)      NOT NULL DEFAULT 'api',
    created_at    timestamptz      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_predictions_unit ON monitoring.predictions (unit_id, cycle);

-- Показания датчиков. В raw нет ограничений уникальности: сюда пишется всё, что пришло
-- из файла или из потока, а дубли и ошибки отсеиваются при переходе в cleaned.
CREATE TABLE IF NOT EXISTS raw.sensor_readings (
    id         bigserial PRIMARY KEY,
    dataset    varchar(16)  NOT NULL,           -- train / test / stream
    unit_id    integer,                         -- номер двигателя
    cycle      integer,                         -- номер цикла работы
    setting_1  double precision,                -- режимные параметры
    setting_2  double precision,
    setting_3  double precision,
    s_1  double precision, s_2  double precision, s_3  double precision, s_4  double precision,
    s_5  double precision, s_6  double precision, s_7  double precision, s_8  double precision,
    s_9  double precision, s_10 double precision, s_11 double precision, s_12 double precision,
    s_13 double precision, s_14 double precision, s_15 double precision, s_16 double precision,
    s_17 double precision, s_18 double precision, s_19 double precision, s_20 double precision,
    s_21 double precision,
    source     varchar(128) NOT NULL,           -- имя файла или топика
    loaded_at  timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_raw_sensor_readings_unit
    ON raw.sensor_readings (dataset, unit_id, cycle);

-- Сообщения из Kafka (dataset = 'stream'): раздел и смещение сообщения. Kafka гарантирует доставку
-- «хотя бы один раз», поэтому после сбоя сообщение может прийти повторно; уникальный индекс не даёт
-- записать его дважды (INSERT ... ON CONFLICT DO NOTHING).
ALTER TABLE raw.sensor_readings ADD COLUMN IF NOT EXISTS kafka_partition integer;
ALTER TABLE raw.sensor_readings ADD COLUMN IF NOT EXISTS kafka_offset bigint;
CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_stream_message
    ON raw.sensor_readings (source, kafka_partition, kafka_offset) WHERE kafka_offset IS NOT NULL;

-- Сообщения из Kafka, не прошедшие проверку (битый JSON, ошибки валидации): поток не останавливается,
-- а сообщение сохраняется для разбора.
CREATE TABLE IF NOT EXISTS raw.stream_rejects (
    id              bigserial PRIMARY KEY,
    source          varchar(128) NOT NULL,
    kafka_partition integer      NOT NULL,
    kafka_offset    bigint       NOT NULL,
    payload         text,
    error           text         NOT NULL,
    received_at     timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (source, kafka_partition, kafka_offset)
);

-- Истинный остаточный ресурс тестовых двигателей на последнем наблюдаемом цикле (RUL_FD001.txt).
CREATE TABLE IF NOT EXISTS raw.rul_test (
    id         bigserial PRIMARY KEY,
    unit_id    integer,
    rul        integer,
    source     varchar(128) NOT NULL,
    loaded_at  timestamptz  NOT NULL DEFAULT now()
);
