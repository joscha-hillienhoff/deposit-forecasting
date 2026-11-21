"""Rolling-origin backtest: forecast table and per-horizon scores for any model.

``run_backtest`` produces one long table of out-of-sample forecasts, ``score`` aggregates it
into MAE, MAPE and band coverage per model, segment and horizon. See ``docs/methodology.md``
for the design decisions.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

from deposit_forecasting.data import load_panel
from deposit_forecasting.gbm import GradientBoostingForecaster
from deposit_forecasting.models import (
    DEFAULT_LEVEL,
    TARGET,
    DriftForecaster,
    Forecaster,
    NaiveForecaster,
)

MAX_HORIZON = 12
DEFAULT_FIRST_ORIGIN = pd.Period("2010-12", freq="M")
Window = tuple[pd.Period, pd.Period] | None  # inclusive target-month window; None = everything
RATE_TURN: Window = (pd.Period("2022-01", freq="M"), pd.Period("2023-12", freq="M"))
DEFAULT_SEGMENTS: dict[str, Window] = {"full": None, "rate_turn": RATE_TURN}
FORECAST_COLUMNS = ["model", "origin", "horizon", "target", "actual", "mean", "lower", "upper"]
SCORE_COLUMNS = ["model", "segment", "horizon", "n", "mae", "mape", "coverage", "nominal"]
DEFAULT_MODELS: dict[str, Callable[[], Forecaster]] = {
    "naive": NaiveForecaster,
    "drift": DriftForecaster,
    "lightgbm": GradientBoostingForecaster,
}
TABLE_HORIZONS = (1, 3, 6, 12)


def run_backtest(
    panel: pd.DataFrame,
    models: Mapping[str, Callable[[], Forecaster]],
    *,
    first_origin: pd.Period | str = DEFAULT_FIRST_ORIGIN,
    max_horizon: int = MAX_HORIZON,
) -> pd.DataFrame:
    """Expanding-window rolling-origin forecasts for every model.

    One origin per month from ``first_origin`` to the last month that still has a realised
    month after it. At each origin a fresh model is built from its factory and fitted on the
    panel up to and including the origin, so it never sees later rows. The realised driver
    columns (all columns except the target) of the forecast months are passed as ``exog``:
    an ex-post evaluation conditional on the realised driver path. Near the end of the panel
    only the months that are realised are forecast, so fewer horizons are scored.

    Returns a long table with the columns ``FORECAST_COLUMNS``.
    """
    first = pd.Period(first_origin, freq="M")
    if first not in panel.index:
        raise ValueError(f"first origin {first} is not in the panel")
    drivers = panel.drop(columns=TARGET)
    rows: list[pd.DataFrame] = []
    for origin in panel.index[panel.index >= first][:-1]:
        history = panel.loc[:origin]
        future = panel.loc[origin + 1 :].iloc[:max_horizon]
        for name, factory in models.items():
            model = factory().fit(history)
            forecast = model.predict(len(future), exog=drivers.loc[future.index])
            rows.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "origin": origin,
                        "horizon": range(1, len(future) + 1),
                        "target": future.index,
                        "actual": future[TARGET].to_numpy(),
                        "mean": forecast["mean"].to_numpy(),
                        "lower": forecast["lower"].to_numpy(),
                        "upper": forecast["upper"].to_numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)[FORECAST_COLUMNS]


def score(
    forecasts: pd.DataFrame,
    *,
    segments: Mapping[str, Window] = DEFAULT_SEGMENTS,
    level: float = DEFAULT_LEVEL,
) -> pd.DataFrame:
    """MAE (EUR bn), MAPE (%), band coverage and count per model, segment and horizon.

    A segment is an inclusive window of target months (``None`` for all). ``nominal`` is the
    band level the models claim, to be read next to the observed ``coverage``.
    """
    rows = []
    for segment, window in segments.items():
        part = forecasts
        if window is not None:
            part = forecasts[
                (forecasts["target"] >= window[0]) & (forecasts["target"] <= window[1])
            ]
        error = (part["mean"] - part["actual"]).abs()
        part = part.assign(
            abs_error=error,
            pct_error=100 * error / part["actual"].abs(),
            covered=(part["lower"] <= part["actual"]) & (part["actual"] <= part["upper"]),
        )
        grouped = part.groupby(["model", "horizon"], sort=True)
        result = grouped.agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            mape=("pct_error", "mean"),
            coverage=("covered", "mean"),
        ).reset_index()
        result.insert(1, "segment", segment)
        result["nominal"] = level
        rows.append(result)
    return pd.concat(rows, ignore_index=True)[SCORE_COLUMNS]


def format_table(scores: pd.DataFrame, horizons: tuple[int, ...] = TABLE_HORIZONS) -> str:
    """Compact text table of the scores at selected horizons."""
    shown = scores[scores["horizon"].isin(horizons)].copy()
    shown["mape"] = shown["mape"].round(2)
    shown["mae"] = shown["mae"].round(2)
    shown["coverage"] = shown["coverage"].round(2)
    return shown.to_string(index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="deposit-forecasting-backtest",
        description="Run the baselines through the rolling-origin backtest and print the scores.",
    )
    parser.add_argument(
        "--first-origin",
        default=str(DEFAULT_FIRST_ORIGIN),
        help="first forecast origin, YYYY-MM (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, help="directory for forecasts.csv and scores.csv (created if needed)"
    )
    args = parser.parse_args(argv)
    panel = load_panel()
    forecasts = run_backtest(panel, DEFAULT_MODELS, first_origin=args.first_origin)
    scores = score(forecasts)
    print(format_table(scores))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(args.out / "forecasts.csv", index=False)
        scores.to_csv(args.out / "scores.csv", index=False)


if __name__ == "__main__":
    main()
