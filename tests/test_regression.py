import numpy as np
import pandas as pd
import pytest

from deposit_forecasting import scenarios
from deposit_forecasting.backtest import FORECAST_COLUMNS, run_backtest
from deposit_forecasting.regression import FEATURES, RateRegressionForecaster
from deposit_forecasting.scenarios import (
    SCENARIO_COLUMNS,
    estimate_pass_through,
    plot_scenarios,
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


def test_rate_paths_recursion_with_known_coefficients():
    index = pd.period_range("2020-01", periods=3, freq="M")
    panel = pd.DataFrame({"overnight_rate": 1.0, "time_deposit_rate": 2.0}, index=index)
    known = pd.DataFrame(
        {"phi": [0.5, 0.0], "impact": [0.1, 0.4]},
        index=["overnight_rate", "time_deposit_rate"],
    )
    up = rate_paths(panel, 100, horizon=3, pass_through=known)
    # delta_h = phi * delta_{h-1} + impact * shift: overnight 0.1, 0.15, 0.175; deposit rate 0.4
    np.testing.assert_allclose(up["overnight_rate"], [1.1, 1.15, 1.175])
    np.testing.assert_allclose(up["time_deposit_rate"], [2.4, 2.4, 2.4])
    np.testing.assert_allclose(up["spread"], [1.3, 1.25, 1.225])
    down = rate_paths(panel, -100, horizon=3, pass_through=known)
    np.testing.assert_allclose(down["overnight_rate"], [0.9, 0.85, 0.825])
    np.testing.assert_allclose(down["time_deposit_rate"], [1.6, 1.6, 1.6])
    flat = rate_paths(panel, 0, horizon=3, pass_through=known)
    np.testing.assert_allclose(flat["spread"], [1.0, 1.0, 1.0])


def test_rate_regression_runs_through_backtest(panel):
    history = panel.iloc[:320]
    first = history.index[-14]
    table = run_backtest(history, {"rate_regression": RateRegressionForecaster}, first_origin=first)
    assert list(table.columns) == FORECAST_COLUMNS
    assert set(table["model"]) == {"rate_regression"}
    # the final origins have fewer than 12 realised future months, so exog is shorter
    per_origin = table.groupby("origin")["horizon"].max()
    assert per_origin.iloc[0] == 12 and per_origin.iloc[-1] == 1
    # an origin with the full window reproduces fit + predict on the realised spread path
    origin = per_origin.index[0]
    future = history.loc[origin + 1 : origin + 12, ["spread"]]
    direct = RateRegressionForecaster().fit(history.loc[:origin]).predict(12, exog=future)
    got = table[table["origin"] == origin].set_index("target")
    np.testing.assert_allclose(got["mean"], direct["mean"])
    # a truncated final window is accepted and forecasts only the available months
    last = table[table["origin"] == per_origin.index[-1]]
    assert list(last["target"]) == [history.index[-1]]
    # the model is correctly specified here, so a correct handoff forecasts accurately
    assert ((table["mean"] / table["actual"] - 1).abs() < 0.05).all()


def test_plot_scenarios_writes_png(panel, tmp_path):
    table = run_scenarios(panel, horizon=6)
    path = tmp_path / "figs" / "scenarios.png"
    plot_scenarios(panel, table, path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_scenarios_cli(panel, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scenarios, "load_panel", lambda: panel)
    out = tmp_path / "chart.png"
    scenarios.main(["--out", str(out)])
    printed = capsys.readouterr().out
    assert all(name in printed for name in ["baseline", "+100bp", "-100bp", "time_deposit_rate"])
    assert out.read_bytes().startswith(b"\x89PNG")
