-- Слои enriched и features.
-- ФАЙЛ СГЕНЕРИРОВАН функцией src.data.features.build_layers_sql — правьте генератор, а не этот файл.
-- Перегенерировать: .\.venv\Scripts\python.exe -m src.data.features
-- Таблицы пересоздаются целиком, скрипт можно запускать повторно.

-- ============ enriched: остаточный ресурс (RUL) и целевая переменная ============
-- train: двигатель наблюдается до отказа, RUL = последний цикл - текущий цикл.
-- test:  наблюдение обрывается раньше; RUL на последнем цикле известен из RUL_FD001.txt,
--        RUL = истинный RUL + (последний наблюдаемый цикл - текущий цикл).
-- stream и другие источники: RUL неизвестен (NULL).
-- CASCADE удаляет и функции слоя features (они пересоздаются ниже).
DROP TABLE IF EXISTS features.engine_features CASCADE;
DROP TABLE IF EXISTS enriched.sensor_readings;

CREATE TABLE enriched.sensor_readings AS
WITH last_cycle AS (
    SELECT dataset, unit_id, max(cycle) AS max_cycle
    FROM cleaned.sensor_readings
    GROUP BY dataset, unit_id
), with_rul AS (
    SELECT c.*,
           l.max_cycle,
           (c.cycle = l.max_cycle) AS is_last_cycle,
           CASE c.dataset
               WHEN 'train' THEN l.max_cycle - c.cycle
               WHEN 'test'  THEN r.rul + l.max_cycle - c.cycle
           END AS rul
    FROM cleaned.sensor_readings c
    JOIN last_cycle l USING (dataset, unit_id)
    LEFT JOIN cleaned.rul_test r ON c.dataset = 'test' AND r.unit_id = c.unit_id
)
SELECT *,
       CASE WHEN rul IS NULL THEN NULL WHEN rul <= 30 THEN 1 ELSE 0 END::smallint AS target
FROM with_rul;

ALTER TABLE enriched.sensor_readings ADD PRIMARY KEY (dataset, unit_id, cycle);

-- ============ features: признаки по истории двигателя ============
-- Для каждого датчика: текущее значение, среднее и стандартное отклонение за 10 циклов,
-- наклон тренда за 20 циклов, отклонение среднего от «нормы» (средняя за первые
-- 10 циклов). Окна включают только текущий и предыдущие циклы: утечки из будущего нет.
-- Таблица создаётся пустой и заполняется функцией backfill_engine_features: расчёт описан один раз.
CREATE TABLE features.engine_features AS
SELECT
    dataset,
    unit_id,
    cycle,
    s_2,
    avg(s_2) OVER w_mean AS s_2_mean_10,
    coalesce(stddev_samp(s_2) OVER w_mean, 0) AS s_2_std_10,
    coalesce(regr_slope(s_2, cycle) OVER w_slope, 0) AS s_2_slope_20,
    avg(s_2) OVER w_mean - avg(s_2) FILTER (WHERE cycle <= 10) OVER w_all AS s_2_delta_base,
    s_3,
    avg(s_3) OVER w_mean AS s_3_mean_10,
    coalesce(stddev_samp(s_3) OVER w_mean, 0) AS s_3_std_10,
    coalesce(regr_slope(s_3, cycle) OVER w_slope, 0) AS s_3_slope_20,
    avg(s_3) OVER w_mean - avg(s_3) FILTER (WHERE cycle <= 10) OVER w_all AS s_3_delta_base,
    s_4,
    avg(s_4) OVER w_mean AS s_4_mean_10,
    coalesce(stddev_samp(s_4) OVER w_mean, 0) AS s_4_std_10,
    coalesce(regr_slope(s_4, cycle) OVER w_slope, 0) AS s_4_slope_20,
    avg(s_4) OVER w_mean - avg(s_4) FILTER (WHERE cycle <= 10) OVER w_all AS s_4_delta_base,
    s_7,
    avg(s_7) OVER w_mean AS s_7_mean_10,
    coalesce(stddev_samp(s_7) OVER w_mean, 0) AS s_7_std_10,
    coalesce(regr_slope(s_7, cycle) OVER w_slope, 0) AS s_7_slope_20,
    avg(s_7) OVER w_mean - avg(s_7) FILTER (WHERE cycle <= 10) OVER w_all AS s_7_delta_base,
    s_8,
    avg(s_8) OVER w_mean AS s_8_mean_10,
    coalesce(stddev_samp(s_8) OVER w_mean, 0) AS s_8_std_10,
    coalesce(regr_slope(s_8, cycle) OVER w_slope, 0) AS s_8_slope_20,
    avg(s_8) OVER w_mean - avg(s_8) FILTER (WHERE cycle <= 10) OVER w_all AS s_8_delta_base,
    s_9,
    avg(s_9) OVER w_mean AS s_9_mean_10,
    coalesce(stddev_samp(s_9) OVER w_mean, 0) AS s_9_std_10,
    coalesce(regr_slope(s_9, cycle) OVER w_slope, 0) AS s_9_slope_20,
    avg(s_9) OVER w_mean - avg(s_9) FILTER (WHERE cycle <= 10) OVER w_all AS s_9_delta_base,
    s_11,
    avg(s_11) OVER w_mean AS s_11_mean_10,
    coalesce(stddev_samp(s_11) OVER w_mean, 0) AS s_11_std_10,
    coalesce(regr_slope(s_11, cycle) OVER w_slope, 0) AS s_11_slope_20,
    avg(s_11) OVER w_mean - avg(s_11) FILTER (WHERE cycle <= 10) OVER w_all AS s_11_delta_base,
    s_12,
    avg(s_12) OVER w_mean AS s_12_mean_10,
    coalesce(stddev_samp(s_12) OVER w_mean, 0) AS s_12_std_10,
    coalesce(regr_slope(s_12, cycle) OVER w_slope, 0) AS s_12_slope_20,
    avg(s_12) OVER w_mean - avg(s_12) FILTER (WHERE cycle <= 10) OVER w_all AS s_12_delta_base,
    s_13,
    avg(s_13) OVER w_mean AS s_13_mean_10,
    coalesce(stddev_samp(s_13) OVER w_mean, 0) AS s_13_std_10,
    coalesce(regr_slope(s_13, cycle) OVER w_slope, 0) AS s_13_slope_20,
    avg(s_13) OVER w_mean - avg(s_13) FILTER (WHERE cycle <= 10) OVER w_all AS s_13_delta_base,
    s_14,
    avg(s_14) OVER w_mean AS s_14_mean_10,
    coalesce(stddev_samp(s_14) OVER w_mean, 0) AS s_14_std_10,
    coalesce(regr_slope(s_14, cycle) OVER w_slope, 0) AS s_14_slope_20,
    avg(s_14) OVER w_mean - avg(s_14) FILTER (WHERE cycle <= 10) OVER w_all AS s_14_delta_base,
    s_15,
    avg(s_15) OVER w_mean AS s_15_mean_10,
    coalesce(stddev_samp(s_15) OVER w_mean, 0) AS s_15_std_10,
    coalesce(regr_slope(s_15, cycle) OVER w_slope, 0) AS s_15_slope_20,
    avg(s_15) OVER w_mean - avg(s_15) FILTER (WHERE cycle <= 10) OVER w_all AS s_15_delta_base,
    s_17,
    avg(s_17) OVER w_mean AS s_17_mean_10,
    coalesce(stddev_samp(s_17) OVER w_mean, 0) AS s_17_std_10,
    coalesce(regr_slope(s_17, cycle) OVER w_slope, 0) AS s_17_slope_20,
    avg(s_17) OVER w_mean - avg(s_17) FILTER (WHERE cycle <= 10) OVER w_all AS s_17_delta_base,
    s_20,
    avg(s_20) OVER w_mean AS s_20_mean_10,
    coalesce(stddev_samp(s_20) OVER w_mean, 0) AS s_20_std_10,
    coalesce(regr_slope(s_20, cycle) OVER w_slope, 0) AS s_20_slope_20,
    avg(s_20) OVER w_mean - avg(s_20) FILTER (WHERE cycle <= 10) OVER w_all AS s_20_delta_base,
    s_21,
    avg(s_21) OVER w_mean AS s_21_mean_10,
    coalesce(stddev_samp(s_21) OVER w_mean, 0) AS s_21_std_10,
    coalesce(regr_slope(s_21, cycle) OVER w_slope, 0) AS s_21_slope_20,
    avg(s_21) OVER w_mean - avg(s_21) FILTER (WHERE cycle <= 10) OVER w_all AS s_21_delta_base,
    rul,
    target,
    is_last_cycle
FROM enriched.sensor_readings
WHERE false
WINDOW
    w_mean  AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
    w_slope AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
    w_all   AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);

ALTER TABLE features.engine_features ADD PRIMARY KEY (dataset, unit_id, cycle);

-- Заполнение (дозаполнение) исторических признаков из enriched для выбранных наборов и двигателей.
-- NULL в параметре означает «все». Строки выбранных двигателей пересчитываются целиком, поэтому
-- функцию можно вызывать повторно, например после прихода новых данных из потока.
-- Фильтр отбирает двигатели целиком, а окна считаются внутри двигателя — значения не зависят от фильтра.
CREATE OR REPLACE FUNCTION features.backfill_engine_features(
    p_dataset  text      DEFAULT NULL,
    p_unit_ids integer[] DEFAULT NULL
) RETURNS integer LANGUAGE plpgsql AS $$
DECLARE
    n integer;
BEGIN
    DELETE FROM features.engine_features
    WHERE (p_dataset IS NULL OR dataset = p_dataset) AND (p_unit_ids IS NULL OR unit_id = ANY(p_unit_ids));
    INSERT INTO features.engine_features
    SELECT
        dataset,
        unit_id,
        cycle,
        s_2,
        avg(s_2) OVER w_mean AS s_2_mean_10,
        coalesce(stddev_samp(s_2) OVER w_mean, 0) AS s_2_std_10,
        coalesce(regr_slope(s_2, cycle) OVER w_slope, 0) AS s_2_slope_20,
        avg(s_2) OVER w_mean - avg(s_2) FILTER (WHERE cycle <= 10) OVER w_all AS s_2_delta_base,
        s_3,
        avg(s_3) OVER w_mean AS s_3_mean_10,
        coalesce(stddev_samp(s_3) OVER w_mean, 0) AS s_3_std_10,
        coalesce(regr_slope(s_3, cycle) OVER w_slope, 0) AS s_3_slope_20,
        avg(s_3) OVER w_mean - avg(s_3) FILTER (WHERE cycle <= 10) OVER w_all AS s_3_delta_base,
        s_4,
        avg(s_4) OVER w_mean AS s_4_mean_10,
        coalesce(stddev_samp(s_4) OVER w_mean, 0) AS s_4_std_10,
        coalesce(regr_slope(s_4, cycle) OVER w_slope, 0) AS s_4_slope_20,
        avg(s_4) OVER w_mean - avg(s_4) FILTER (WHERE cycle <= 10) OVER w_all AS s_4_delta_base,
        s_7,
        avg(s_7) OVER w_mean AS s_7_mean_10,
        coalesce(stddev_samp(s_7) OVER w_mean, 0) AS s_7_std_10,
        coalesce(regr_slope(s_7, cycle) OVER w_slope, 0) AS s_7_slope_20,
        avg(s_7) OVER w_mean - avg(s_7) FILTER (WHERE cycle <= 10) OVER w_all AS s_7_delta_base,
        s_8,
        avg(s_8) OVER w_mean AS s_8_mean_10,
        coalesce(stddev_samp(s_8) OVER w_mean, 0) AS s_8_std_10,
        coalesce(regr_slope(s_8, cycle) OVER w_slope, 0) AS s_8_slope_20,
        avg(s_8) OVER w_mean - avg(s_8) FILTER (WHERE cycle <= 10) OVER w_all AS s_8_delta_base,
        s_9,
        avg(s_9) OVER w_mean AS s_9_mean_10,
        coalesce(stddev_samp(s_9) OVER w_mean, 0) AS s_9_std_10,
        coalesce(regr_slope(s_9, cycle) OVER w_slope, 0) AS s_9_slope_20,
        avg(s_9) OVER w_mean - avg(s_9) FILTER (WHERE cycle <= 10) OVER w_all AS s_9_delta_base,
        s_11,
        avg(s_11) OVER w_mean AS s_11_mean_10,
        coalesce(stddev_samp(s_11) OVER w_mean, 0) AS s_11_std_10,
        coalesce(regr_slope(s_11, cycle) OVER w_slope, 0) AS s_11_slope_20,
        avg(s_11) OVER w_mean - avg(s_11) FILTER (WHERE cycle <= 10) OVER w_all AS s_11_delta_base,
        s_12,
        avg(s_12) OVER w_mean AS s_12_mean_10,
        coalesce(stddev_samp(s_12) OVER w_mean, 0) AS s_12_std_10,
        coalesce(regr_slope(s_12, cycle) OVER w_slope, 0) AS s_12_slope_20,
        avg(s_12) OVER w_mean - avg(s_12) FILTER (WHERE cycle <= 10) OVER w_all AS s_12_delta_base,
        s_13,
        avg(s_13) OVER w_mean AS s_13_mean_10,
        coalesce(stddev_samp(s_13) OVER w_mean, 0) AS s_13_std_10,
        coalesce(regr_slope(s_13, cycle) OVER w_slope, 0) AS s_13_slope_20,
        avg(s_13) OVER w_mean - avg(s_13) FILTER (WHERE cycle <= 10) OVER w_all AS s_13_delta_base,
        s_14,
        avg(s_14) OVER w_mean AS s_14_mean_10,
        coalesce(stddev_samp(s_14) OVER w_mean, 0) AS s_14_std_10,
        coalesce(regr_slope(s_14, cycle) OVER w_slope, 0) AS s_14_slope_20,
        avg(s_14) OVER w_mean - avg(s_14) FILTER (WHERE cycle <= 10) OVER w_all AS s_14_delta_base,
        s_15,
        avg(s_15) OVER w_mean AS s_15_mean_10,
        coalesce(stddev_samp(s_15) OVER w_mean, 0) AS s_15_std_10,
        coalesce(regr_slope(s_15, cycle) OVER w_slope, 0) AS s_15_slope_20,
        avg(s_15) OVER w_mean - avg(s_15) FILTER (WHERE cycle <= 10) OVER w_all AS s_15_delta_base,
        s_17,
        avg(s_17) OVER w_mean AS s_17_mean_10,
        coalesce(stddev_samp(s_17) OVER w_mean, 0) AS s_17_std_10,
        coalesce(regr_slope(s_17, cycle) OVER w_slope, 0) AS s_17_slope_20,
        avg(s_17) OVER w_mean - avg(s_17) FILTER (WHERE cycle <= 10) OVER w_all AS s_17_delta_base,
        s_20,
        avg(s_20) OVER w_mean AS s_20_mean_10,
        coalesce(stddev_samp(s_20) OVER w_mean, 0) AS s_20_std_10,
        coalesce(regr_slope(s_20, cycle) OVER w_slope, 0) AS s_20_slope_20,
        avg(s_20) OVER w_mean - avg(s_20) FILTER (WHERE cycle <= 10) OVER w_all AS s_20_delta_base,
        s_21,
        avg(s_21) OVER w_mean AS s_21_mean_10,
        coalesce(stddev_samp(s_21) OVER w_mean, 0) AS s_21_std_10,
        coalesce(regr_slope(s_21, cycle) OVER w_slope, 0) AS s_21_slope_20,
        avg(s_21) OVER w_mean - avg(s_21) FILTER (WHERE cycle <= 10) OVER w_all AS s_21_delta_base,
        rul,
        target,
        is_last_cycle
    FROM enriched.sensor_readings
    WHERE (p_dataset IS NULL OR dataset = p_dataset) AND (p_unit_ids IS NULL OR unit_id = ANY(p_unit_ids))
    WINDOW
        w_mean  AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
        w_slope AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        w_all   AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);
    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n;
END
$$;

-- Признаки двигателя «на момент» цикла: последняя строка с cycle <= p_as_of_cycle
-- (без параметра — последнее известное состояние). Используется для выборок на прошлую дату.
CREATE OR REPLACE FUNCTION features.get_engine_features(
    p_dataset     text,
    p_unit_id     integer,
    p_as_of_cycle integer DEFAULT NULL
) RETURNS SETOF features.engine_features LANGUAGE sql STABLE AS $$
    SELECT * FROM features.engine_features
    WHERE dataset = p_dataset AND unit_id = p_unit_id AND (p_as_of_cycle IS NULL OR cycle <= p_as_of_cycle)
    ORDER BY cycle DESC
    LIMIT 1
$$;

SELECT features.backfill_engine_features();
