import numpy as np
import pytest

from src.models.metrics import engine_alerts, evaluate, false_positive_rate, threshold_for_recall


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
