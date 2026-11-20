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


def last_row_features(history: list[dict]) -> dict:
    """Признаки только для последнего цикла двигателя — быстрый вариант для онлайн-режима.

    history — показания одного двигателя, отсортированные по циклу. Достаточно хранить первые
    BASELINE_CYCLES циклов («норма») и последние SLOPE_WINDOW циклов: результат совпадает
    с compute_features для последней строки полной истории.
    """
    cycles = np.array([h["cycle"] for h in history], dtype=float)
    out = {"cycle": int(cycles[-1])}
    base_mask = cycles <= BASELINE_CYCLES
    xs = cycles[-SLOPE_WINDOW:]
    n = len(xs)
    sx, sxx = xs.sum(), (xs * xs).sum()
    den = n * sxx - sx * sx
    for s in FEATURE_SENSORS:
        y = np.array([h[s] for h in history], dtype=float)
        win = y[-MEAN_WINDOW:]
        mean = win.mean()
        ys = y[-SLOPE_WINDOW:]
        slope = (n * (xs * ys).sum() - sx * ys.sum()) / den if den != 0 else 0.0
        out[s] = float(y[-1])
        out[f"{s}_mean_{MEAN_WINDOW}"] = float(mean)
        out[f"{s}_std_{MEAN_WINDOW}"] = float(win.std(ddof=1)) if len(win) >= 2 else 0.0
        out[f"{s}_slope_{SLOPE_WINDOW}"] = float(slope)
        out[f"{s}_delta_base"] = float(mean - y[base_mask].mean())
    return out


def make_target(rul: pd.Series) -> pd.Series:
    """Целевая переменная: 1, если до отказа осталось HORIZON циклов или меньше."""
    return (rul <= HORIZON).astype(int)


def build_layers_sql() -> str:
    """Генерирует SQL для слоёв enriched и features и функций слоя features."""
    return f"""-- Слои enriched и features.
-- ФАЙЛ СГЕНЕРИРОВАН функцией src.data.features.build_layers_sql — правьте генератор, а не этот файл.
-- Перегенерировать: .\\.venv\\Scripts\\python.exe -m src.data.features
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
       CASE WHEN rul IS NULL THEN NULL WHEN rul <= {HORIZON} THEN 1 ELSE 0 END::smallint AS target
FROM with_rul;

ALTER TABLE enriched.sensor_readings ADD PRIMARY KEY (dataset, unit_id, cycle);

-- ============ features: признаки по истории двигателя ============
-- Для каждого датчика: текущее значение, среднее и стандартное отклонение за {MEAN_WINDOW} циклов,
-- наклон тренда за {SLOPE_WINDOW} циклов, отклонение среднего от «нормы» (средняя за первые
-- {BASELINE_CYCLES} циклов). Окна включают только текущий и предыдущие циклы: утечки из будущего нет.
-- Таблица создаётся пустой и заполняется функцией backfill_engine_features: расчёт описан один раз.
CREATE TABLE features.engine_features AS
{_select_features("WHERE false")};

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
{_select_features("WHERE (p_dataset IS NULL OR dataset = p_dataset) AND (p_unit_ids IS NULL OR unit_id = ANY(p_unit_ids))", indent="    ")};
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
"""


def _select_features(where: str, indent: str = "") -> str:
    """SELECT признаков из enriched с заданным фильтром (общий для создания таблицы и функции)."""
    lines = []
    for s in FEATURE_SENSORS:
        lines += [
            f"    {s},",
            f"    avg({s}) OVER w_mean AS {s}_mean_{MEAN_WINDOW},",
            f"    coalesce(stddev_samp({s}) OVER w_mean, 0) AS {s}_std_{MEAN_WINDOW},",
            f"    coalesce(regr_slope({s}, cycle) OVER w_slope, 0) AS {s}_slope_{SLOPE_WINDOW},",
            f"    avg({s}) OVER w_mean - avg({s}) FILTER (WHERE cycle <= {BASELINE_CYCLES}) OVER w_all"
            f" AS {s}_delta_base,",
        ]
    body = "\n".join([
        "SELECT", "    dataset,", "    unit_id,", "    cycle,", *lines,
        "    rul,", "    target,", "    is_last_cycle",
        "FROM enriched.sensor_readings", where, "WINDOW",
        f"    w_mean  AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN {MEAN_WINDOW - 1} PRECEDING AND CURRENT ROW),",
        f"    w_slope AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN {SLOPE_WINDOW - 1} PRECEDING AND CURRENT ROW),",
        "    w_all   AS (PARTITION BY dataset, unit_id ORDER BY cycle ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
    ])
    return "\n".join(indent + line for line in body.splitlines())


LAYERS_SQL_PATH = SQL_DIR / "layers.sql"

if __name__ == "__main__":
    LAYERS_SQL_PATH.write_text(build_layers_sql(), encoding="utf-8")
    print(f"Записан {LAYERS_SQL_PATH}: {len(FEATURE_COLS)} признаков")
