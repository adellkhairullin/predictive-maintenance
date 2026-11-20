import numpy as np
import pandas as pd
import pytest

from src.data.features import (FEATURE_COLS, FEATURE_SENSORS, LAYERS_SQL_PATH, build_layers_sql,
                               compute_features, make_target)


def make_unit(n_cycles: int, unit_id: int = 1, dataset: str = "train", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"dataset": dataset, "unit_id": unit_id, "cycle": np.arange(1, n_cycles + 1)})
    for s in FEATURE_SENSORS:
        df[s] = rng.normal(100, 1, n_cycles)
    return df


def test_output_columns_and_no_nans():
    out = compute_features(make_unit(30))
    assert list(out.columns) == ["dataset", "unit_id"] + FEATURE_COLS
    assert not out[FEATURE_COLS].isna().any().any()


def test_linear_sensor_gives_exact_slope_and_zero_std_on_first_cycle():
    df = make_unit(25)
    df["s_2"] = 5.0 + 0.5 * df["cycle"]
    out = compute_features(df)
    assert out["s_2_std_10"].iloc[0] == 0.0
    assert out["s_2_slope_20"].iloc[0] == 0.0          # по одной точке тренд не определён
    assert np.allclose(out["s_2_slope_20"].iloc[1:], 0.5)
    assert out["s_2_mean_10"].iloc[9] == pytest.approx(df["s_2"].iloc[:10].mean())


def test_delta_base_uses_first_cycles_as_norm():
    df = make_unit(20)
    df["s_3"] = 10.0
    df.loc[df["cycle"] > 10, "s_3"] = 20.0
    out = compute_features(df)
    assert (out["s_3_delta_base"].iloc[:10] == 0).all()
    assert out["s_3_delta_base"].iloc[-1] == pytest.approx(10.0)  # среднее за 11..20 = 20, норма = 10


def test_no_future_leakage():
    df = make_unit(40)
    changed = df.copy()
    changed.loc[changed["cycle"] > 25, FEATURE_SENSORS] += 1000   # меняем «будущее»
    a, b = compute_features(df), compute_features(changed)
    pd.testing.assert_frame_equal(a[a["cycle"] <= 25], b[b["cycle"] <= 25])


def test_units_are_independent_and_order_does_not_matter():
    u1, u2 = make_unit(15, unit_id=1, seed=1), make_unit(15, unit_id=2, seed=2)
    together = compute_features(pd.concat([u2, u1]).sample(frac=1, random_state=42))
    alone = compute_features(u1)
    pd.testing.assert_frame_equal(together[together["unit_id"] == 1].reset_index(drop=True), alone)


def test_make_target_horizon():
    assert make_target(pd.Series([0, 30, 31, 100])).tolist() == [1, 1, 0, 0]


def test_layers_sql_is_up_to_date():
    assert LAYERS_SQL_PATH.read_text(encoding="utf-8") == build_layers_sql(), \
        "sql/layers.sql устарел: выполните python -m src.data.features"
