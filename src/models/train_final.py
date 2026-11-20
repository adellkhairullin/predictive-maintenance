"""Финальная модель v2 (блок 3.3): логистическая регрессия с порогом «с запасом».

Выбор алгоритма — по результатам блока 3.2 (notebooks/03_model_selection.ipynb).
1. Порог: кросс-валидация по двигателям (10 блоков) на train, наибольший порог, при котором нижняя
   граница 95% бутстрап-ДИ Recall >= 0,85.
2. Модель обучается на всём train.
3. Единственная проверка на test: метрики, бизнес-метрики по двигателям, гипотеза H3
   (бутстрап по тестовым двигателям: нижняя граница Recall >= 0,85 и верхняя граница FPR < 0,15).
4. Модель и метаданные (порог, признаки, метрики, версии библиотек, отпечаток данных) —
   в models/final_v2.joblib и models/final_v2.json.

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m src.models.train_final
"""
from __future__ import annotations

import hashlib
import json

from src.data.features import FEATURE_COLS, HORIZON
from src.data.load_raw import FILES, RAW_DIR, RUL_FILE
from src.models.data import load_features, split_xy
from src.models.evaluation import N_SPLITS, cross_validate_oof, single_row_latency_ms
from src.models.metrics import (MAX_FPR, MIN_RECALL, bootstrap_ci, engine_alerts, evaluate,
                                threshold_bootstrap_recall)
from src.models.registry import save_model
from src.models.train_baseline import make_baseline

MODEL_NAME = "final_v2"


def data_fingerprint() -> dict:
    """MD5 исходных файлов: по нему видно, на каких именно данных обучена модель."""
    files = list(FILES.values()) + [RUL_FILE]
    return {f: hashlib.md5((RAW_DIR / f).read_bytes()).hexdigest() for f in files}


def run(train, test, features: list[str] = FEATURE_COLS, save: bool = True) -> dict:
    X, y, groups = split_xy(train, features)
    X_test, y_test, g_test = split_xy(test, features)

    model = make_baseline()
    oof, folds = cross_validate_oof(model, X, y, groups, n_splits=N_SPLITS)
    thr = threshold_bootstrap_recall(y, oof, groups, MIN_RECALL)
    threshold = thr["threshold"]

    model.fit(X, y)
    proba = model.predict_proba(X_test)[:, 1]
    h3 = bootstrap_ci(y_test, proba, g_test, threshold)
    h3["confirmed"] = bool(h3["recall"]["lo"] >= MIN_RECALL and h3["fpr"]["hi"] < MAX_FPR)

    result = {
        "model": MODEL_NAME, "version": "v2", "algorithm": "LogisticRegression",
        "params": model.named_steps["clf"].get_params(),
        "features": features, "horizon_cycles": HORIZON,
        "threshold": threshold, "threshold_info": thr,
        "threshold_rule": f"max threshold with lower bound of 95% bootstrap CI (by engines) "
                          f"of OOF recall >= {MIN_RECALL}",
        "cv": {"n_splits": N_SPLITS, "pooled_oof": evaluate(y, oof, threshold),
               "fold_pr_auc": folds["pr_auc"].round(6).tolist(),
               "pr_auc_mean": float(folds["pr_auc"].mean()), "pr_auc_std": float(folds["pr_auc"].std()),
               "engine_level": engine_alerts(groups, train["rul"], oof, threshold, HORIZON)},
        "test": evaluate(y_test, proba, threshold),
        "test_engine_level": engine_alerts(g_test, test["rul"], proba, threshold, HORIZON),
        "h3": h3,
        "latency_single_row": single_row_latency_ms(model, X_test, n=1000),
        "data": {"train_rows": int(len(train)), "test_rows": int(len(test)),
                 "train_engines": int(groups.nunique()), "md5": data_fingerprint()},
    }
    if save:
        save_model(model, MODEL_NAME, result)
    return {**result, "_model": model, "_oof": oof, "_test_proba": proba}


if __name__ == "__main__":
    train, test = load_features()
    res = run(train, test)
    show = {k: res[k] for k in ["threshold", "test", "test_engine_level", "h3", "latency_single_row"]}
    print(json.dumps(show, ensure_ascii=False, indent=2))
