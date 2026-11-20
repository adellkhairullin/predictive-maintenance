"""Baseline-модель v1: логистическая регрессия.

1. Кросс-валидация по двигателям (10 блоков) на train -> предсказания вне обучения.
2. Порог подбирается по этим предсказаниям под Recall >= 0,85.
3. Модель обучается на всём train и проверяется на test с тем же порогом.
4. Модель и метаданные сохраняются в models/baseline_v1.*

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m src.models.train_baseline
"""
from __future__ import annotations

import json

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import FEATURE_COLS, HORIZON
from src.models.data import load_features, split_xy
from src.models.evaluation import N_SPLITS, cross_validate_oof, single_row_latency_ms
from src.models.metrics import MIN_RECALL, engine_alerts, evaluate, threshold_for_recall
from src.models.registry import save_model

RANDOM_STATE = 42
MODEL_NAME = "baseline_v1"


def make_baseline() -> Pipeline:
    """Масштабирование признаков + логистическая регрессия с балансировкой классов."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                                   random_state=RANDOM_STATE)),
    ])


def run(train, test, features: list[str] = FEATURE_COLS, save: bool = True) -> dict:
    X, y, groups = split_xy(train, features)
    X_test, y_test, _ = split_xy(test, features)

    model = make_baseline()
    oof, folds = cross_validate_oof(model, X, y, groups)
    threshold = threshold_for_recall(y, oof, MIN_RECALL)
    cv_metrics = evaluate(y, oof, threshold)

    model.fit(X, y)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold)
    latency = single_row_latency_ms(model, X_test)

    result = {
        "model": MODEL_NAME, "version": "v1", "algorithm": "LogisticRegression",
        "params": model.named_steps["clf"].get_params(),
        "features": features, "horizon_cycles": HORIZON,
        "threshold": threshold, "threshold_rule": f"max threshold with OOF recall >= {MIN_RECALL}",
        "cv": {"n_splits": N_SPLITS, "scheme": "GroupKFold by unit_id",
               "pooled_oof": cv_metrics,
               "fold_pr_auc": folds["pr_auc"].round(6).tolist(),
               "pr_auc_mean": float(folds["pr_auc"].mean()), "pr_auc_std": float(folds["pr_auc"].std())},
        "test": test_metrics,
        "engine_level": {
            "cv_oof": engine_alerts(train["unit_id"], train["rul"], oof, threshold, HORIZON),
            "test": engine_alerts(test["unit_id"], test["rul"], test_proba, threshold, HORIZON),
        },
        "latency_single_row": latency,
    }
    if save:
        save_model(model, MODEL_NAME, result)
    return {**result, "_model": model, "_oof": oof, "_folds": folds, "_test_proba": test_proba}


if __name__ == "__main__":
    train, test = load_features()
    res = run(train, test)
    show = {k: res[k] for k in ["threshold", "cv", "test", "engine_level", "latency_single_row"]}
    print(json.dumps(show, ensure_ascii=False, indent=2))
