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

-- 7. Контроль дублей ключа в raw (для потоковых данных; ожидается пустой результат для файлов)
SELECT dataset, unit_id, cycle, count(*) AS copies
FROM raw.sensor_readings
GROUP BY dataset, unit_id, cycle
HAVING count(*) > 1
ORDER BY copies DESC
LIMIT 20;
