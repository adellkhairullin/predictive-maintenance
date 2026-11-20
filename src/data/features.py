"""Признаки по истории двигателя: слои enriched и features.

Один и тот же расчёт реализован дважды:
* в SQL (sql/layers.sql, генерируется функцией build_layers_sql) — для пакетной обработки в PostgreSQL;
* на pandas (compute_features) — для онлайн-предсказаний в API и потоке.
Тест tests/test_features_db.py проверяет, что обе версии дают одинаковый результат.

Все окна причинные: используют только текущий и предыдущие циклы двигателя, поэтому
информация из будущего в признаки не попадает.

Перегенерировать sql/layers.sql после изменения параметров:
    .\\.venv\\Scripts\\python.exe -m src.data.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.db import SQL_DIR

# Датчики, которые остались после очистки (см. notebooks/01_data_preparation.ipynb).
# setting_1 и setting_2 не используются: с износом не связаны (|rho| < 0,01).
FEATURE_SENSORS = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11",
                   "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
MEAN_WINDOW = 10      # окно для среднего и разброса, циклов
SLOPE_WINDOW = 20     # окно для тренда, циклов
BASELINE_CYCLES = 10  # первые циклы двигателя, которые считаются «нормой»
HORIZON = 30          # горизонт прогноза: target = 1, если RUL <= 30

GROUP = ["dataset", "unit_id"]
KEY = GROUP + ["cycle"]
SUFFIXES = [f"mean_{MEAN_WINDOW}", f"std_{MEAN_WINDOW}", f"slope_{SLOPE_WINDOW}", "delta_base"]
FEATURE_COLS = ["cycle"] + [c for s in FEATURE_SENSORS for c in [s] + [f"{s}_{x}" for x in SUFFIXES]]
LABEL_COLS = ["rul", "target", "is_last_cycle"]


def _unit_features(g: pd.DataFrame) -> pd.DataFrame:
    """Признаки для одного двигателя (строки отсортированы по циклу)."""
    x = g["cycle"].astype(float)
    out = {}
    for s in FEATURE_SENSORS:
        y = g[s].astype(float)
        mean = y.rolling(MEAN_WINDOW, min_periods=1).mean()
        std = y.rolling(MEAN_WINDOW, min_periods=2).std(ddof=1).fillna(0.0)

        r = lambda v: v.rolling(SLOPE_WINDOW, min_periods=1).sum()
        n, sx, sy, sxy, sxx = r(pd.Series(1.0, index=y.index)), r(x), r(y), r(x * y), r(x * x)
        den = n * sxx - sx * sx
        slope = ((n * sxy - sx * sy) / den.where(den != 0)).fillna(0.0)

        base = y.where(g["cycle"] <= BASELINE_CYCLES).expanding().mean()

        out[s] = y
        out[f"{s}_mean_{MEAN_WINDOW}"] = mean
        out[f"{s}_std_{MEAN_WINDOW}"] = std
        out[f"{s}_slope_{SLOPE_WINDOW}"] = slope
        out[f"{s}_delta_base"] = mean - base
    return pd.DataFrame(out, index=g.index)


def compute_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Считает признаки на pandas. На входе — очищенные показания (dataset, unit_id, cycle, датчики)."""
    df = readings.sort_values(KEY).reset_index(drop=True)
    feats = pd.concat([_unit_features(g) for _, g in df.groupby(GROUP, sort=False)])
    return pd.concat([df[KEY], feats.loc[df.index]], axis=1)


def make_target(rul: pd.Series) -> pd.Series:
    """Целевая переменная: 1, если до отказа осталось HORIZON циклов или меньше."""
    return (rul <= HORIZON).astype(int)


def build_layers_sql() -> str:
    """Генерирует SQL для слоёв enriched и features."""
    feature_lines = []
    for s in FEATURE_SENSORS:
        feature_lines += [
            f"    {s},",
            f"    avg({s}) OVER w_mean AS {s}_mean_{MEAN_WINDOW},",
            f"    coalesce(stddev_samp({s}) OVER w_mean, 0) AS {s}_std_{MEAN_WINDOW},",
            f"    coalesce(regr_slope({s}, cycle) OVER w_slope, 0) AS {s}_slope_{SLOPE_WINDOW},",
            f"    avg({s}) OVER w_mean - avg({s}) FILTER (WHERE cycle <= {BASELINE_CYCLES}) OVER w_all"
            f" AS {s}_delta_base,",
        ]
    features_sql = "\n".join(feature_lines)
    return f"""-- Слои enriched и features.
-- ФАЙЛ СГЕНЕРИРОВАН функцией src.data.features.build_layers_sql — правьте генератор, а не этот файл.
-- Перегенерировать: .\\.venv\\Scripts\\python.exe -m src.data.features
-- Таблицы пересоздаются целиком, скрипт можно запускать повторно.

-- ============ enriched: остаточный ресурс (RUL) и целевая переменная ============
-- train: двигатель наблюдается до отказа, RUL = последний цикл - текущий цикл.
-- test:  наблюдение обрывается раньше; RUL на последнем цикле известен из RUL_FD001.txt,
--        RUL = истинный RUL + (последний наблюдаемый цикл - текущий цикл).
-- stream и другие источники: RUL неизвестен (NULL).
DROP TABLE IF EXISTS features.engine_features;
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
       CASE WHEN rul IS NULL THEN NULL WHEN rul <= {HORIZON} THEN 1 ELSE 0 END::smallint AS target
FROM with_rul;

ALTER TABLE enriched.sensor_readings ADD PRIMARY KEY (dataset, unit_id, cycle);

-- ============ features: признаки по истории двигателя ============
-- Для каждого датчика: текущее значение, среднее и стандартное отклонение за {MEAN_WINDOW} циклов,
-- наклон тренда за {SLOPE_WINDOW} циклов, отклонение среднего от «нормы» (средняя за первые
-- {BASELINE_CYCLES} циклов). Окна включают только текущий и предыдущие циклы: утечки из будущего нет.
CREATE TABLE features.engine_features AS
SELECT
    dataset,
    unit_id,
    cycle,
{features_sql}
    rul,
    target,
    is_last_cycle
FROM enriched.sensor_readings
WINDOW
    w_mean  AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN {MEAN_WINDOW - 1} PRECEDING AND CURRENT ROW),
    w_slope AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN {SLOPE_WINDOW - 1} PRECEDING AND CURRENT ROW),
    w_all   AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);

ALTER TABLE features.engine_features ADD PRIMARY KEY (dataset, unit_id, cycle);
"""


LAYERS_SQL_PATH = SQL_DIR / "layers.sql"

if __name__ == "__main__":
    LAYERS_SQL_PATH.write_text(build_layers_sql(), encoding="utf-8")
    print(f"Записан {LAYERS_SQL_PATH}: {len(FEATURE_COLS)} признаков")
