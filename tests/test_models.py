import numpy as np
import pandas as pd

from src.models.evaluation import cross_validate_oof, single_row_latency_ms
from src.models.registry import load_model, save_model
from src.models.train_baseline import make_baseline, run


def synthetic(n_units: int = 20, cycles: int = 60, seed: int = 42) -> pd.DataFrame:
    """Двигатели с линейным износом: признак растёт к отказу, target = RUL <= 10."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        rul = np.arange(cycles - 1, -1, -1)
        rows.append(pd.DataFrame({"unit_id": u, "cycle": np.arange(1, cycles + 1), "rul": rul,
                                  "f1": -rul / cycles + rng.normal(0, 0.05, cycles),
                                  "f2": rng.normal(size=cycles),
                                  "target": (rul <= 10).astype(int), "is_last_cycle": rul == 0}))
    return pd.concat(rows, ignore_index=True)


def test_cv_blocks_do_not_mix_engines():
    df = synthetic()
    oof, folds = cross_validate_oof(make_baseline(), df[["f1", "f2"]], df["target"], df["unit_id"],
                                    n_splits=5)
    assert len(oof) == len(df) and ((oof >= 0) & (oof <= 1)).all()
    assert folds["units"].sum() == df["unit_id"].nunique()  # каждый двигатель ровно в одном блоке
    assert folds["pr_auc"].mean() > 0.9


def test_run_produces_threshold_meeting_recall_on_cv():
    train, test = synthetic(seed=1), synthetic(n_units=10, seed=2)
    res = run(train, test, features=["f1", "f2"], save=False)
    assert res["cv"]["pooled_oof"]["recall"] >= 0.85
    assert 0 < res["threshold"] < 1
    assert len(res["cv"]["fold_pr_auc"]) == 10
    assert res["engine_level"]["cv_oof"]["engines"] == 20


def test_save_and_load_roundtrip(tmp_path):
    df = synthetic(n_units=4)
    model = make_baseline().fit(df[["f1", "f2"]], df["target"])
    save_model(model, "tmp_model", {"threshold": 0.5, "features": ["f1", "f2"]}, models_dir=tmp_path)
    loaded, meta = load_model("tmp_model", models_dir=tmp_path)
    np.testing.assert_allclose(loaded.predict_proba(df[["f1", "f2"]]), model.predict_proba(df[["f1", "f2"]]))
    assert meta["threshold"] == 0.5 and "scikit-learn" in meta["libraries"]


def test_candidate_search_spaces_are_valid():
    """Каждую конфигурацию из пространства поиска можно применить к модели, и модель обучается."""
    from sklearn.base import clone
    from sklearn.model_selection import ParameterSampler
    from src.models.train_candidates import candidates

    specs = candidates()
    assert set(specs) == {"logistic_regression_tuned", "random_forest", "hist_gradient_boosting", "mlp"}
    df = synthetic(n_units=4)
    for name, spec in specs.items():
        params = next(iter(ParameterSampler(spec["space"], n_iter=1, random_state=42)))
        model = clone(spec["estimator"]).set_params(**params)
        if name == "random_forest":
            model.set_params(n_estimators=10)
        model.fit(df[["f1", "f2"]], df["target"])
        assert model.predict_proba(df[["f1", "f2"]]).shape == (len(df), 2)


def test_latency_measurement():
    df = synthetic(n_units=2)
    model = make_baseline().fit(df[["f1", "f2"]], df["target"])
    lat = single_row_latency_ms(model, df[["f1", "f2"]], n=30, warmup=5)
    assert lat["n"] == 30 and 0 < lat["p50_ms"] <= lat["p95_ms"]
