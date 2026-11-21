import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deposit_forecasting import figures

MODELS = list(figures.COLORS)


def make_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(
        [
            {
                "model": m,
                "segment": seg,
                "horizon": h,
                "n": 10,
                "mae": float(h),
                "mape": 1.0,
                "coverage": 0.8,
                "nominal": 0.8,
            }
            for m in MODELS
            for seg in figures.SEGMENT_TITLES
            for h in (1, 2, 3)
        ]
    )
    origin = figures.TURN_ORIGIN
    targets = pd.period_range(origin + 1, periods=3, freq="M")
    forecasts = pd.DataFrame(
        {
            "model": [m for m in MODELS for _ in targets],
            "origin": origin,
            "horizon": list(range(1, 4)) * len(MODELS),
            "target": list(targets) * len(MODELS),
            "mean": 100.0,
        }
    )
    return scores, forecasts


def test_figures_are_written(tmp_path):
    scores, forecasts = make_results()
    actual = pd.Series(100.0, index=pd.period_range("2019-01", "2022-12", freq="M"))
    paths = [
        figures.plot_mae_by_horizon(scores, tmp_path / "mae.png"),
        figures.plot_coverage_by_horizon(scores, tmp_path / "coverage.png"),
        figures.plot_turn_forecasts(forecasts, actual, tmp_path / "turn.png"),
    ]
    for path in paths:
        image = plt.imread(path)
        assert image.shape[0] > 100 and image.shape[1] > 100
        assert len(np.unique(image.reshape(-1, image.shape[-1]), axis=0)) > 10


def test_load_results_parses_periods(tmp_path):
    scores, forecasts = make_results()
    scores.to_csv(tmp_path / "backtest_scores.csv", index=False)
    forecasts.to_csv(tmp_path / "backtest_forecasts.csv", index=False)
    loaded_scores, loaded_forecasts = figures.load_results(tmp_path)
    assert len(loaded_scores) == len(scores)
    assert loaded_scores.loc[0, "coverage"] == 0.8
    assert list(loaded_forecasts["target"].iloc[:3]) == list(
        pd.period_range("2022-01", periods=3, freq="M")
    )
    assert (loaded_forecasts["origin"] == figures.TURN_ORIGIN).all()
    assert (loaded_forecasts["mean"] == 100.0).all()


def test_every_default_model_has_a_colour():
    from deposit_forecasting.backtest import DEFAULT_MODELS

    assert set(DEFAULT_MODELS) == set(figures.COLORS) == set(figures.LABELS)


def test_summary_table_has_one_row_per_model():
    scores, _ = make_results()
    table = figures.summary_table(scores, horizons=(1, 3))
    assert list(table.index) == list(figures.LABELS.values())
    assert table.shape[1] == 2 * 2 * 2  # segments x metrics x horizons
    assert table.loc["Naive", ("full", "coverage", 1)] == 80.0
    assert table.loc["Naive", ("rate_turn", "coverage", 3)] == 80.0
    assert table.loc["Naive", ("full", "mae", 1)] == 1.0
    assert table.loc["LightGBM", ("rate_turn", "mae", 3)] == 3.0


def test_summary_table_pivots_each_cell_from_its_own_row():
    scores, _ = make_results()
    mask = (
        (scores["model"] == "ets") & (scores["segment"] == "rate_turn") & (scores["horizon"] == 3)
    )
    scores.loc[mask, ["mae", "coverage"]] = [42.0, 0.25]
    table = figures.summary_table(scores, horizons=(1, 3))
    assert table.loc["ETS", ("rate_turn", "mae", 3)] == 42.0
    assert table.loc["ETS", ("rate_turn", "coverage", 3)] == 25.0
    assert table.loc["ETS", ("full", "mae", 3)] == 3.0
    assert table.loc["SARIMA", ("rate_turn", "mae", 3)] == 3.0
