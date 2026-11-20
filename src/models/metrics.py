"""Метрики качества и подбор порога классификации под бизнес-требования."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

MIN_RECALL = 0.85   # требование: Recall >= 0,85
MAX_FPR = 0.15      # требование: FPR < 0,15


def false_positive_rate(y_true, y_pred) -> float:
    """Доля ложных тревог среди исправных состояний: FP / (FP + TN)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


def threshold_for_recall(y_true, proba, min_recall: float = MIN_RECALL) -> float:
    """Наибольший порог, при котором Recall >= min_recall.

    Чем выше порог, тем меньше ложных тревог, поэтому берём максимальный порог,
    который ещё выполняет требование по Recall. Предсказание: 1, если вероятность >= порога.
    """
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = recall[:-1] >= min_recall   # recall[i] соответствует thresholds[i]
    if not ok.any():
        return float(np.min(proba))
    return float(thresholds[ok].max())


def threshold_bootstrap_recall(y_true, proba, units, min_recall: float = MIN_RECALL,
                               n_boot: int = 1000, ci: float = 0.95,
                               random_state: int = 42) -> dict:
    """Порог «с запасом»: наибольший порог, при котором НИЖНЯЯ граница доверительного
    интервала Recall (бутстрап по двигателям) не меньше min_recall.

    Бутстрап выбирает двигатели с возвращением, а не отдельные строки, потому что циклы одного
    двигателя зависимы. Recall зависит только от положительных примеров, поэтому для каждого
    двигателя достаточно знать вероятности на его положительных строках.
    """
    y_true, proba, units = np.asarray(y_true), np.asarray(proba), np.asarray(units)
    pos = y_true == 1
    eng = np.unique(units[pos])
    cand = np.unique(proba[pos])                       # возможные пороги
    # count_ge[e, j] — число положительных строк двигателя e с вероятностью >= cand[j]
    count_ge = np.zeros((len(eng), len(cand)))
    n_pos = np.zeros(len(eng))
    for i, e in enumerate(eng):
        p = np.sort(proba[pos & (units == e)])
        n_pos[i] = len(p)
        count_ge[i] = len(p) - np.searchsorted(p, cand, side="left")
    rng = np.random.default_rng(random_state)
    weights = rng.multinomial(len(eng), np.full(len(eng), 1 / len(eng)), size=n_boot)
    recall = (weights @ count_ge) / (weights @ n_pos)[:, None]       # n_boot x пороги
    lower = np.quantile(recall, (1 - ci) / 2, axis=0)
    ok = lower >= min_recall
    j = np.flatnonzero(ok).max() if ok.any() else 0
    return {"threshold": float(cand[j]), "recall_ci_lower": float(lower[j]),
            "recall_ci_upper": float(np.quantile(recall[:, j], 1 - (1 - ci) / 2)),
            "n_boot": n_boot, "engines": int(len(eng))}


def evaluate(y_true, proba, threshold: float) -> dict:
    """Метрики, не зависящие от порога (PR-AUC, ROC-AUC), и метрики при заданном пороге."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    rec, fpr = recall_score(y_true, y_pred, zero_division=0), false_positive_rate(y_true, y_pred)
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "threshold": float(threshold),
        "recall": float(rec),
        "fpr": float(fpr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "meets_recall": bool(rec >= MIN_RECALL),
        "meets_fpr": bool(fpr < MAX_FPR),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def engine_alerts(units, rul, proba, threshold: float, horizon: int = 30) -> dict:
    """Бизнес-метрики уровня двигателя в окне последних horizon циклов перед отказом.

    * detected_share — доля двигателей, по которым в окне была хотя бы одна тревога
      («доля отказов, о которых предупредили заранее»);
    * median_lead — медиана запаса времени: RUL на первой тревоге в окне, циклов.
    Учитываются только двигатели, у которых есть наблюдения в окне.
    """
    df = pd.DataFrame({"unit": np.asarray(units), "rul": np.asarray(rul),
                       "alarm": np.asarray(proba) >= threshold})
    win = df[df["rul"] <= horizon]
    per_unit = win.groupby("unit").apply(
        lambda g: pd.Series({"detected": bool(g["alarm"].any()),
                             "lead": g.loc[g["alarm"], "rul"].max() if g["alarm"].any() else np.nan}),
        include_groups=False)
    leads = per_unit["lead"].dropna()
    return {
        "engines": int(len(per_unit)),
        "detected_share": float(per_unit["detected"].mean()) if len(per_unit) else float("nan"),
        "median_lead": float(leads.median()) if len(leads) else float("nan"),
        "min_lead": float(leads.min()) if len(leads) else float("nan"),
    }
