"""Кросс-валидация по двигателям и замер времени предсказания."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

N_SPLITS = 10  # 10 блоков: при 5 парных оценках критерий Уилкоксона не может дать p < 0,05


def cross_validate_oof(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                       n_splits: int = N_SPLITS) -> tuple[np.ndarray, pd.DataFrame]:
    """Кросс-валидация GroupKFold: все циклы одного двигателя попадают в один блок.

    Возвращает предсказания вне обучения (out-of-fold) для каждой строки и метрики по блокам.
    Разбиение детерминированное, поэтому у всех моделей одинаковые блоки (нужно для парных сравнений).
    """
    oof = np.zeros(len(y))
    rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        rows.append({"fold": fold, "units": int(groups.iloc[va].nunique()),
                     "pr_auc": average_precision_score(y.iloc[va], p),
                     "roc_auc": roc_auc_score(y.iloc[va], p)})
    return oof, pd.DataFrame(rows)


def single_row_latency_ms(model, X: pd.DataFrame, n: int = 500, warmup: int = 20) -> dict:
    """Время предсказания для одной строки (как в онлайн-режиме), миллисекунды."""
    rows = [X.iloc[[i % len(X)]] for i in range(n + warmup)]
    times = []
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        model.predict_proba(row)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "n": n}
