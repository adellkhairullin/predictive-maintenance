-- Аналитические и контрольные запросы к слоям данных.
-- Выполнять после python -m src.etl_pipeline. Каждый запрос самостоятелен.

-- 1. Число строк и двигателей по слоям
SELECT 'raw' AS layer, dataset, count(*) AS rows, count(DISTINCT unit_id) AS units
FROM raw.sensor_readings GROUP BY dataset
UNION ALL
SELECT 'cleaned', dataset, count(*), count(DISTINCT unit_id) FROM cleaned.sensor_readings GROUP BY dataset
UNION ALL
SELECT 'enriched', dataset, count(*), count(DISTINCT unit_id) FROM enriched.sensor_readings GROUP BY dataset
UNION ALL
SELECT 'features', dataset, count(*), count(DISTINCT unit_id) FROM features.engine_features GROUP BY dataset
ORDER BY layer, dataset;

-- 2. Баланс классов целевой переменной (target = 1: до отказа <= 30 циклов)
SELECT dataset, target, count(*) AS rows, round(100.0 * count(*) / sum(count(*)) OVER (PARTITION BY dataset), 2) AS pct
FROM enriched.sensor_readings
GROUP BY dataset, target
ORDER BY dataset, target;

-- 3. Срок жизни двигателей обучающей выборки (циклов до отказа)
SELECT min(max_cycle) AS min_life, round(avg(max_cycle), 1) AS avg_life,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY max_cycle) AS median_life, max(max_cycle) AS max_life
FROM (SELECT DISTINCT unit_id, max_cycle FROM enriched.sensor_readings WHERE dataset = 'train') t;

-- 4. Средние показания ключевых датчиков по этапам жизни двигателя
SELECT CASE WHEN rul <= 30 THEN '1) <= 30 до отказа'
            WHEN rul <= 100 THEN '2) 31-100'
            ELSE '3) > 100' END AS stage,
       count(*) AS rows,
       round(avg(s_4)::numeric, 2)  AS s_4_avg,
       round(avg(s_11)::numeric, 3) AS s_11_avg,
       round(avg(s_12)::numeric, 2) AS s_12_avg
FROM enriched.sensor_readings
WHERE dataset = 'train'
GROUP BY stage
ORDER BY stage;

-- 5. Последнее состояние тестовых двигателей: истинный RUL и тренд ключевого датчика
SELECT unit_id, cycle AS last_cycle, rul, target,
       round(s_11_slope_20::numeric, 4) AS s_11_trend,
       round(s_11_delta_base::numeric, 3) AS s_11_vs_norm
FROM features.engine_features
WHERE dataset = 'test' AND is_last_cycle
ORDER BY rul
LIMIT 10;

-- 6. Контроль качества: пропуски в признаках и некорректный RUL (ожидается 0 и 0)
SELECT count(*) FILTER (WHERE s_2_mean_10 IS NULL OR s_11_slope_20 IS NULL OR s_4_delta_base IS NULL) AS null_features,
       count(*) FILTER (WHERE dataset IN ('train', 'test') AND (rul IS NULL OR rul < 0))     AS bad_rul
FROM features.engine_features;

-- ==================== Мониторинг (по журналу monitoring.predictions) ====================

-- 8. Задержка и объём прогнозов по источникам за последние сутки (API и Kafka)
SELECT source,
       count(*)                                                                AS predictions,
       round(percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p50_ms,
       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p95_ms,
       round(100.0 * avg(alarm::int), 2)                                       AS alarm_pct
FROM monitoring.predictions
WHERE created_at > now() - interval '1 day'
GROUP BY source
ORDER BY source;

-- 9. Дрейф входных данных: доля прогнозов с показаниями вне диапазона обучающих данных, по датчикам
SELECT split_part(w, ':', 2) AS sensor,
       count(*)              AS out_of_range,
       round(100.0 * count(*) / (SELECT count(*) FROM monitoring.predictions), 3) AS pct_of_predictions
FROM monitoring.predictions, unnest(warnings) AS w
WHERE w LIKE 'out_of_range:%'
GROUP BY sensor
ORDER BY out_of_range DESC;

-- 10. Качество на известных исходах: прогнозы потока по тестовым двигателям (номер = исходный + смещение
--     producer, по умолчанию 1000) сопоставляются с фактическим RUL из enriched -> Recall и FPR
WITH scored AS (
    SELECT p.alarm, e.target
    FROM monitoring.predictions p
    JOIN enriched.sensor_readings e
      ON e.dataset = 'test' AND e.unit_id = p.unit_id - 1000 AND e.cycle = p.cycle
    WHERE p.source = 'kafka' AND p.unit_id BETWEEN 1001 AND 1100
)
SELECT count(*)                                                                    AS scored,
       round(avg(alarm::int) FILTER (WHERE target = 1)::numeric, 3)                AS recall,
       round(avg(alarm::int) FILTER (WHERE target = 0)::numeric, 4)                AS fpr
FROM scored;

-- 11. Контроль дублей ключа в raw (для потоковых данных; ожидается пустой результат для файлов)
SELECT dataset, unit_id, cycle, count(*) AS copies
FROM raw.sensor_readings
GROUP BY dataset, unit_id, cycle
HAVING count(*) > 1
ORDER BY copies DESC
LIMIT 20;
