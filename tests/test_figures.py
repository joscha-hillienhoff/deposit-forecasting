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
    assert all(p.stat().st_size > 0 for p in paths)


def test_load_results_parses_periods(tmp_path):
    scores, forecasts = make_results()
    scores.to_csv(tmp_path / "backtest_scores.csv", index=False)
    forecasts.to_csv(tmp_path / "backtest_forecasts.csv", index=False)
    loaded_scores, loaded_forecasts = figures.load_results(tmp_path)
    assert len(loaded_scores) == len(scores)
    assert loaded_forecasts["target"].iloc[0] == pd.Period("2022-01", freq="M")


def test_every_default_model_has_a_colour():
    from deposit_forecasting.backtest import DEFAULT_MODELS

    assert set(DEFAULT_MODELS) == set(figures.COLORS) == set(figures.LABELS)


def test_summary_table_has_one_row_per_model():
    scores, _ = make_results()
    table = figures.summary_table(scores, horizons=(1, 3))
    assert list(table.index) == list(figures.LABELS.values())
    assert table.shape[1] == 2 * 2 * 2  # segments x metrics x horizons
