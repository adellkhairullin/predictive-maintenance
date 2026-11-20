import pandas as pd
import pytest

from src.models.predict import FailurePredictor
from src.models.registry import MODELS_DIR, save_model
from src.models.train_baseline import make_baseline
from tests.test_models import synthetic


@pytest.fixture
def predictor(tmp_path):
    df = synthetic(n_units=6)
    model = make_baseline().fit(df[["f1", "f2"]], df["target"])
    save_model(model, "toy", {"version": "toy-1", "threshold": 0.5, "features": ["f1", "f2"]},
               models_dir=tmp_path)
    return FailurePredictor("toy", models_dir=tmp_path)


def test_predict_returns_probability_and_alarm(predictor):
    out = predictor.predict(pd.DataFrame({"f1": [-1.0, 0.0], "f2": [0.0, 0.0], "extra": [1, 2]}))
    assert list(out.columns) == ["probability", "alarm"]
    assert out["probability"].between(0, 1).all()
    assert out["alarm"].tolist() == (out["probability"] >= 0.5).tolist()
    assert out["probability"].iloc[1] > out["probability"].iloc[0]   # ближе к отказу — выше вероятность


def test_predict_one(predictor):
    r = predictor.predict_one({"f1": 0.0, "f2": 0.1})
    assert set(r) == {"probability", "alarm", "threshold", "model_version"}
    assert r["model_version"] == "toy-1" and r["threshold"] == 0.5


def test_missing_features_raise(predictor):
    with pytest.raises(ValueError, match="Нет признаков"):
        predictor.predict(pd.DataFrame({"f1": [0.0]}))


@pytest.mark.skipif(not (MODELS_DIR / "final_v2.joblib").exists(), reason="финальная модель не обучена")
def test_final_model_file_loads():
    p = FailurePredictor("final_v2")
    assert p.version == "v2" and len(p.features) == 71 and 0 < p.threshold < 1
