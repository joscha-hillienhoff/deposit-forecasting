import numpy as np
import pandas as pd
import pytest

from deposit_forecasting.regression import FEATURES, RateRegressionForecaster
from deposit_forecasting.scenarios import (
    SCENARIO_COLUMNS,
    estimate_pass_through,
    rate_paths,
    run_scenarios,
)

TRUE = {"const": 0.004, "spread_lag1": -0.01, "dlog_lag1": 0.3}


def synthetic_panel(n: int = 400, seed: int = 0, noise: float = 0.002) -> pd.DataFrame:
    """Deposits generated from the regression itself, plus a persistent rate system."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("1990-01", periods=n, freq="M")
    policy = np.cumsum(rng.normal(0, 0.1, n)) + 2.0
    overnight = 0.3 * policy + rng.normal(0, 0.02, n)
    time_deposit = policy + rng.normal(0, 0.05, n)
    spread = time_deposit - overnight
    season = {m: 0.002 * np.sin(m) for m in range(1, 13)}
    change = np.zeros(n)
    for t in range(2, n):
        change[t] = (
            TRUE["const"]
            + TRUE["spread_lag1"] * spread[t - 1]
            + TRUE["dlog_lag1"] * change[t - 1]
            + season[index[t].month]
            + rng.normal(0, noise)
        )
    return pd.DataFrame(
        {
            "deposits": 100 * np.exp(np.cumsum(change)),
            "overnight_rate": overnight,
            "time_deposit_rate": time_deposit,
            "spread": spread,
            "deposit_facility_rate": policy,
        },
        index=index,
    )


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return synthetic_panel()


def test_interface_contract_and_determinism(panel):
    history = panel.iloc[:300]
    exog = panel.iloc[300:312].drop(columns="deposits")
    out = RateRegressionForecaster().fit(history).predict(12, exog=exog)
    assert list(out.columns) == ["mean", "lower", "upper"]
    assert isinstance(out.index, pd.PeriodIndex)
    assert out.index[0] == history.index[-1] + 1 and len(out) == 12
    assert (out.lower <= out["mean"]).all() and (out["mean"] <= out.upper).all()
    width = out.upper - out.lower
    assert width.is_monotonic_increasing and width.iloc[-1] > width.iloc[0]
    again = RateRegressionForecaster().fit(history).predict(12, exog=exog)
    pd.testing.assert_frame_equal(out, again)


def test_coefficients_recover_known_values(panel):
    coef = RateRegressionForecaster().fit(panel).coefficients()
    assert list(coef.index) == FEATURES
    assert list(coef.columns) == ["estimate", "std_error", "t_value"]
    for name, value in TRUE.items():
        # Season dummies are relative to January, which shifts the constant by its effect.
        tolerance = 0.003 if name == "const" else 0.002 if name == "spread_lag1" else 0.08
        assert coef.loc[name, "estimate"] == pytest.approx(value, abs=tolerance)


def test_missing_exog_raises(panel):
    model = RateRegressionForecaster().fit(panel.iloc[:300])
    with pytest.raises(ValueError, match="exog"):
        model.predict(3)
    with pytest.raises(ValueError, match="exog"):
        model.predict(3, exog=panel.iloc[300:303][["deposit_facility_rate"]])
    with pytest.raises(ValueError, match="exog"):
        model.predict(3, exog=panel.iloc[300:302][["spread"]])
    with pytest.raises(RuntimeError):
        RateRegressionForecaster().predict(3, exog=panel.iloc[300:303][["spread"]])


def test_fit_needs_spread(panel):
    with pytest.raises(ValueError, match="spread"):
        RateRegressionForecaster().fit(panel[["deposits"]])


def test_higher_spread_lowers_forecast(panel):
    model = RateRegressionForecaster().fit(panel)
    future = pd.period_range(panel.index[-1] + 1, periods=12, freq="M")
    low = model.predict(12, exog=pd.DataFrame({"spread": 0.5}, index=future))
    high = model.predict(12, exog=pd.DataFrame({"spread": 2.5}, index=future))
    assert (high["mean"].iloc[1:] < low["mean"].iloc[1:]).all()


def test_scenarios_on_synthetic_data(panel):
    table = run_scenarios(panel, horizon=24)
    assert list(table.columns) == SCENARIO_COLUMNS
    assert len(table) == 3 * 24
    assert list(table["scenario"].unique()) == ["baseline", "+100bp", "-100bp"]
    by = {name: part.set_index("month") for name, part in table.groupby("scenario")}
    assert by["baseline"]["spread"].nunique() == 1
    assert (by["+100bp"]["spread"] > by["baseline"]["spread"]).all()
    assert (by["-100bp"]["spread"] < by["baseline"]["spread"]).all()
    assert (by["+100bp"]["mean"].iloc[1:] < by["baseline"]["mean"].iloc[1:]).all()
    assert (by["-100bp"]["mean"].iloc[1:] > by["baseline"]["mean"].iloc[1:]).all()


def test_pass_through_is_asymmetric(panel):
    pt = estimate_pass_through(panel)
    assert pt.loc["time_deposit_rate", "long_run"] > pt.loc["overnight_rate", "long_run"]
    paths = rate_paths(panel, 100, horizon=6)
    assert paths.index[0] == panel.index[-1] + 1 and len(paths) == 6
    base = rate_paths(panel, 0, horizon=6)
    assert (base["time_deposit_rate"] == panel["time_deposit_rate"].iloc[-1]).all()
