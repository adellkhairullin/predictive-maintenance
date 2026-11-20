"""Подбор гиперпараметров и сравнение алгоритмов (блок 3.2).

Порядок:
1. Для каждого алгоритма — случайный поиск гиперпараметров (RandomizedSearchCV) с кросс-валидацией
   GroupKFold на 5 блоков по двигателям, метрика PR-AUC.
2. Лучшая конфигурация каждого алгоритма (и baseline) оценивается по общему протоколу: те же
   10 блоков GroupKFold, что у baseline -> предсказания вне обучения (OOF), PR-AUC по блокам,
   порог по нижней границе 95% ДИ Recall (бутстрап по двигателям), метрики при этом пороге.
3. Для гипотезы H1 те же модели оцениваются на исходных показаниях (номер цикла + 14 датчиков).
Тестовая выборка здесь НЕ используется: она нужна только для финальной проверки в блоке 3.3.

Результаты: models/selection/model_selection.json и models/selection/oof_predictions.csv.

Запуск из корня проекта (около 20–40 минут на 4-ядерном ноутбуке):
    .\\.venv\\Scripts\\python.exe -m src.models.train_candidates
"""
from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import FEATURE_COLS, FEATURE_SENSORS, HORIZON
from src.models.data import load_features, split_xy
from src.models.evaluation import N_SPLITS, cross_validate_oof, single_row_latency_ms
from src.models.metrics import MIN_RECALL, engine_alerts, evaluate, threshold_bootstrap_recall
from src.models.registry import MODELS_DIR, library_versions
from src.models.train_baseline import make_baseline

RANDOM_STATE = 42
TUNING_SPLITS = 5
OUT_DIR = MODELS_DIR / "selection"
RAW_FEATURES = ["cycle"] + FEATURE_SENSORS   # исходные показания без признаков по окну (для H1)
log = logging.getLogger("candidates")


def candidates() -> dict:
    """Алгоритм, пространство поиска, число проверяемых конфигураций, параллельность поиска."""
    return {
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "space": {
                "n_estimators": [200, 400],
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 5, 20, 50],
                "max_features": ["sqrt", 0.3],
                "class_weight": [None, "balanced_subsample"],
            },
            "n_iter": 12, "search_jobs": 1,
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=False),
            "space": {
                "learning_rate": [0.03, 0.05, 0.1, 0.2],
                "max_iter": [100, 200, 400],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [20, 50, 100],
                "l2_regularization": [0.0, 0.1, 1.0],
                "class_weight": [None, "balanced"],
            },
            "n_iter": 20, "search_jobs": 1,
        },
        "mlp": {
            "estimator": Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(random_state=RANDOM_STATE, early_stopping=True,
                                      max_iter=300, n_iter_no_change=10)),
            ]),
            "space": {
                "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                "clf__learning_rate_init": [1e-3, 3e-3],
                "clf__batch_size": [128, 256],
            },
            "n_iter": 10, "search_jobs": 4,
        },
    }


def tune(name: str, spec: dict, X, y, groups) -> dict:
    t0 = time.perf_counter()
    search = RandomizedSearchCV(spec["estimator"], spec["space"], n_iter=spec["n_iter"],
                                scoring="average_precision", cv=GroupKFold(n_splits=TUNING_SPLITS),
                                random_state=RANDOM_STATE, n_jobs=spec["search_jobs"], refit=False)
    search.fit(X, y, groups=groups)
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    top = res[["params", "mean_test_score", "std_test_score", "mean_fit_time"]].head(5)
    log.info("  %s: лучший PR-AUC (5 блоков) = %.4f за %.0f с", name, search.best_score_,
             time.perf_counter() - t0)
    return {"best_params": search.best_params_, "tuning_pr_auc": float(search.best_score_),
            "tuning_seconds": round(time.perf_counter() - t0, 1), "n_configs": spec["n_iter"],
            "top5": [{"params": r.params, "pr_auc_mean": round(r.mean_test_score, 5),
                      "pr_auc_std": round(r.std_test_score, 5), "fit_seconds": round(r.mean_fit_time, 2)}
                     for r in top.itertuples()]}


def assess(model, train: pd.DataFrame, features: list[str]) -> tuple[dict, np.ndarray]:
    """Общий протокол оценки на 10 блоках; тест не используется."""
    X, y, groups = split_xy(train, features)
    oof, folds = cross_validate_oof(model, X, y, groups, n_splits=N_SPLITS)
    thr = threshold_bootstrap_recall(y, oof, groups, MIN_RECALL, random_state=RANDOM_STATE)
    fitted = clone(model).fit(X, y)
    return {
        "fold_pr_auc": folds["pr_auc"].round(6).tolist(),
        "pr_auc_mean": float(folds["pr_auc"].mean()), "pr_auc_std": float(folds["pr_auc"].std()),
        "threshold": thr,
        "pooled_oof": evaluate(y, oof, thr["threshold"]),
        "engine_level": engine_alerts(groups, train["rul"], oof, thr["threshold"], HORIZON),
        "latency_single_row": single_row_latency_ms(fitted, X, n=300),
    }, oof


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> dict:
    train, _ = load_features()
    X, y, groups = split_xy(train, FEATURE_COLS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {"protocol": {"tuning": f"RandomizedSearchCV, GroupKFold({TUNING_SPLITS}), average_precision",
                           "assessment": f"GroupKFold({N_SPLITS}) by unit_id, same folds for all models",
                           "threshold": "max threshold with lower bound of 95% bootstrap CI (by engines) "
                                        f"of OOF recall >= {MIN_RECALL}",
                           "test_used": False},
              "libraries": library_versions(), "models": {}, "h1_raw_features": {}}
    oof_table = train[["unit_id", "cycle", "rul", "target"]].copy()

    log.info("Baseline (логистическая регрессия, без подбора)")
    models = {"logistic_regression": make_baseline()}
    result["models"]["logistic_regression"] = {"tuning": None}

    for name, spec in candidates().items():
        log.info("Подбор гиперпараметров: %s (%s конфигураций)", name, spec["n_iter"])
        tuning = tune(name, spec, X, y, groups)
        models[name] = clone(spec["estimator"]).set_params(**tuning["best_params"])
        result["models"][name] = {"tuning": tuning}

    for name, model in models.items():
        log.info("Оценка на %s блоках: %s", N_SPLITS, name)
        assessment, oof = assess(model, train, FEATURE_COLS)
        result["models"][name].update(assessment)
        oof_table[name] = oof
        log.info("  PR-AUC %.4f ± %.4f, порог %.4f, Recall %.3f, FPR %.4f", assessment["pr_auc_mean"],
                 assessment["pr_auc_std"], assessment["threshold"]["threshold"],
                 assessment["pooled_oof"]["recall"], assessment["pooled_oof"]["fpr"])

    for name in ["logistic_regression", "hist_gradient_boosting"]:
        log.info("H1: %s на исходных показаниях", name)
        Xr, yr, gr = split_xy(train, RAW_FEATURES)
        _, folds = cross_validate_oof(models[name], Xr, yr, gr, n_splits=N_SPLITS)
        result["h1_raw_features"][name] = {"features": RAW_FEATURES,
                                           "fold_pr_auc": folds["pr_auc"].round(6).tolist(),
                                           "pr_auc_mean": float(folds["pr_auc"].mean())}
        log.info("  PR-AUC %.4f", folds["pr_auc"].mean())

    (OUT_DIR / "model_selection.json").write_text(
        json.dumps(to_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    oof_table.to_csv(OUT_DIR / "oof_predictions.csv", index=False, float_format="%.6g")
    log.info("Сохранено в %s", OUT_DIR)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    main()
