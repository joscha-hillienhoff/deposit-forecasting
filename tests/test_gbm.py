import numpy as np
import pandas as pd
import pytest

from deposit_forecasting.backtest import DEFAULT_MODELS, run_backtest
from deposit_forecasting.explain import mean_abs_shap, save_summary_plots
from deposit_forecasting.gbm import GradientBoostingForecaster, origin_features

FAST = {"n_estimators": 10}


def make_panel(months: int = 90, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.period_range("2010-01", periods=months, freq="M")
    spread = 0.5 + 0.1 * rng.standard_normal(months).cumsum()
    growth = 0.004 + 0.002 * rng.standard_normal(months) + 0.01 * np.sin(np.arange(months) * 0.5)
    return pd.DataFrame(
        {
            "deposits": 100.0 * np.exp(growth.cumsum()),
            "overnight_rate": 0.1,
            "spread": spread,
            "deposit_facility_rate": np.round(spread, 1),
        },
        index=index,
    )


def fitted(panel=None, horizons: int = 3, **kwargs):
    panel = make_panel() if panel is None else panel
    model = GradientBoostingForecaster(max_horizon=horizons, params=FAST, **kwargs)
    return model.fit(panel.iloc[:-horizons]), panel.iloc[-horizons:]


def test_interface_index_and_ordering():
    model, future = fitted()
    forecast = model.predict(3, exog=future.drop(columns="deposits"))
    assert list(forecast.columns) == ["mean", "lower", "upper"]
    assert forecast.index.equals(future.index)
    assert (forecast["lower"] <= forecast["mean"]).all()
    assert (forecast["mean"] <= forecast["upper"]).all()
    assert np.isfinite(forecast.to_numpy()).all()


def test_one_model_triple_per_horizon():
    model, _ = fitted(horizons=4)
    assert sorted(model.models_) == [1, 2, 3, 4]
    assert all(set(m) == {"lower", "mean", "upper"} for m in model.models_.values())


def test_deterministic_for_fixed_seed():
    a, future = fitted()
    b, _ = fitted()
    exog = future.drop(columns="deposits")
    pd.testing.assert_frame_equal(a.predict(3, exog=exog), b.predict(3, exog=exog))


def test_features_use_only_rows_up_to_origin():
    panel = make_panel()
    changed = panel.copy()
    changed.iloc[60:, :] = changed.iloc[60:, :] * 2.0
    before = origin_features(panel).iloc[:60]
    after = origin_features(changed).iloc[:60]
    pd.testing.assert_frame_equal(before, after)


def test_forecast_ignores_exog_beyond_spread_of_target_months():
    model, future = fitted()
    exog = future.drop(columns="deposits")
    other = exog.assign(overnight_rate=9.9, deposit_facility_rate=9.9)
    pd.testing.assert_frame_equal(model.predict(3, exog=exog), model.predict(3, exog=other))


def test_input_validation():
    model, future = fitted()
    with pytest.raises(ValueError, match="exog"):
        model.predict(3)
    with pytest.raises(ValueError, match="horizon"):
        model.predict(4, exog=future.drop(columns="deposits"))
    with pytest.raises(RuntimeError):
        GradientBoostingForecaster().predict(1)
    with pytest.raises(ValueError, match="observations"):
        GradientBoostingForecaster().fit(make_panel(20))


def test_registered_in_backtest():
    assert DEFAULT_MODELS["lightgbm"] is GradientBoostingForecaster


def test_shap_table_and_plots(tmp_path):
    model, _ = fitted(horizons=3)
    table = mean_abs_shap(model, horizons=(1, 3))
    features = list(model.train_X_[1].columns)
    for h in (1, 3):
        part = table[table["horizon"] == h]
        assert sorted(part["feature"]) == sorted(features)
        assert (part["mean_abs_shap"] >= 0).all()
    paths = save_summary_plots(model, tmp_path, horizons=(1,))
    assert [p.name for p in paths] == ["shap_h1.png"] and paths[0].stat().st_size > 0


def test_training_target_months_align_with_horizon():
    panel = make_panel()
    model = GradientBoostingForecaster(max_horizon=3, params=FAST).fit(panel)
    spread = panel["spread"]
    for h in (1, 2, 3):
        X = model.train_X_[h]
        expected = spread.shift(-h).reindex(X.index)
        np.testing.assert_allclose(X["spread_target"].to_numpy(), expected.to_numpy())
        assert (X["target_month"].to_numpy() == (X.index + h).month).all()
        assert X.index[-1] == panel.index[-1 - h]


def test_constant_growth_is_extrapolated_by_horizon():
    growth = 0.01
    index = pd.period_range("2010-01", periods=60, freq="M")
    panel = pd.DataFrame(
        {
            "deposits": 100.0 * np.exp(growth * np.arange(60)),
            "overnight_rate": 0.1,
            "spread": 0.5 + 0.1 * np.sin(np.arange(60)),
            "deposit_facility_rate": 0.0,
        },
        index=index,
    )
    history, future = panel.iloc[:-3], panel.iloc[-3:]
    forecast = (
        GradientBoostingForecaster(max_horizon=3, params=FAST)
        .fit(history)
        .predict(3, exog=future.drop(columns="deposits"))
    )
    last = history["deposits"].iloc[-1]
    expected = last * np.exp(growth * np.arange(1, 4))
    np.testing.assert_allclose(forecast["mean"].to_numpy(), expected, rtol=1e-3)


def test_runs_through_rolling_origin_backtest():
    panel = make_panel(44)
    first = panel.index[35]
    forecasts = run_backtest(
        panel,
        {"lightgbm": lambda: GradientBoostingForecaster(max_horizon=3, params=FAST)},
        first_origin=first,
        max_horizon=3,
    )
    origins = panel.index[35:-1]
    assert sorted(forecasts["origin"].unique()) == list(origins)
    for origin, part in forecasts.groupby("origin"):
        remaining = len(panel) - 1 - panel.index.get_loc(origin)
        assert list(part["horizon"]) == list(range(1, min(3, remaining) + 1))
        assert list(part["target"]) == list(panel.index[panel.index > origin][: len(part)])
        np.testing.assert_allclose(part["actual"], panel.loc[part["target"], "deposits"])
    assert np.isfinite(forecasts[["mean", "lower", "upper"]].to_numpy()).all()
    assert (forecasts["lower"] <= forecasts["mean"]).all()
    assert (forecasts["mean"] <= forecasts["upper"]).all()
