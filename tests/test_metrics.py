import numpy as np
import pytest

from src.models.metrics import (engine_alerts, evaluate, false_positive_rate,
                                threshold_bootstrap_recall, threshold_for_recall)


def noisy_engines(n_units=40, rows=30, seed=42):
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), rows)
    y = np.tile(np.r_[np.zeros(rows // 2), np.ones(rows - rows // 2)], n_units).astype(int)
    shift = np.repeat(rng.normal(0, 0.15, n_units), rows)     # двигатели различаются между собой
    p = np.clip(0.35 * y + rng.uniform(0, 0.6, len(y)) + shift, 0, 1)
    return y, p, units


def test_bootstrap_threshold_is_more_conservative_than_plain():
    y, p, units = noisy_engines()
    plain = threshold_for_recall(y, p, 0.85)
    boot = threshold_bootstrap_recall(y, p, units, 0.85, n_boot=500)
    assert boot["threshold"] <= plain
    assert boot["recall_ci_lower"] >= 0.85
    recall = ((p >= boot["threshold"]) & (y == 1)).sum() / y.sum()
    assert recall >= 0.85
    assert boot["engines"] == 40


def test_bootstrap_threshold_is_reproducible():
    y, p, units = noisy_engines()
    a = threshold_bootstrap_recall(y, p, units, n_boot=200, random_state=42)
    b = threshold_bootstrap_recall(y, p, units, n_boot=200, random_state=42)
    assert a == b


def test_false_positive_rate():
    assert false_positive_rate([0, 0, 0, 0, 1], [1, 0, 0, 0, 1]) == 0.25
    assert false_positive_rate([1, 1], [1, 0]) == 0.0   # нет отрицательных примеров


@pytest.mark.parametrize("target", [0.5, 0.85, 1.0])
def test_threshold_for_recall_meets_target_and_is_maximal(target):
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 500), 0, 1)
    thr = threshold_for_recall(y, p, target)
    recall = ((p >= thr) & (y == 1)).sum() / y.sum()
    assert recall >= target
    higher = np.unique(p[p > thr])
    if len(higher):  # следующий по величине порог уже не выполняет требование
        nxt = higher.min()
        assert ((p >= nxt) & (y == 1)).sum() / y.sum() < target


def test_evaluate_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
    assert m["pr_auc"] == m["roc_auc"] == m["recall"] == m["precision"] == m["f2"] == 1.0
    assert m["fpr"] == 0.0 and m["meets_recall"] and m["meets_fpr"]
    assert m["positives"] == 2 and m["n"] == 4


def test_error_cost():
    from src.models.metrics import error_cost
    # 1 пропуск (стоимость 10) и 2 ложные тревоги (по 1)
    assert error_cost([1, 1, 0, 0, 0], [0, 1, 1, 1, 0]) == 12.0


def test_bootstrap_ci_contains_point_and_is_reproducible():
    from src.models.metrics import bootstrap_ci
    y, p, units = noisy_engines()
    a = bootstrap_ci(y, p, units, threshold=0.5, n_boot=300)
    b = bootstrap_ci(y, p, units, threshold=0.5, n_boot=300)
    assert a == b
    for m in ["recall", "fpr"]:
        assert 0 <= a[m]["lo"] <= a[m]["point"] <= a[m]["hi"] <= 1
    assert a["engines"] == 40


def test_engine_alerts_detection_and_lead_time():
    # двигатель 1: тревога с RUL=20 и далее; двигатель 2: тревог в окне нет;
    # двигатель 3: нет наблюдений в окне (RUL > 30) — не учитывается
    units = [1, 1, 1, 1, 2, 2, 3]
    rul = [40, 25, 20, 5, 10, 0, 50]
    proba = [0.9, 0.1, 0.8, 0.9, 0.1, 0.2, 0.9]
    r = engine_alerts(units, rul, proba, threshold=0.5, horizon=30)
    assert r["engines"] == 2
    assert r["detected_share"] == 0.5
    assert r["median_lead"] == 20 and r["min_lead"] == 20
