"""Result figures built from the committed backtest tables.

``deposit-forecasting-figures`` reads ``results/backtest_scores.csv`` and
``results/backtest_forecasts.csv`` (written by ``deposit-forecasting-backtest --out results``)
and draws MAE by horizon, band coverage by horizon and the forecasts from an origin just before
the 2022-2023 rate turn against the realised path. Every model has one fixed colour in all
figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from deposit_forecasting.models import TARGET  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_FIGURE_DIR = ROOT / "docs" / "figures"
SOURCE_NOTE = "Source: ECB Data Portal; own calculations"
TURN_ORIGIN = pd.Period("2021-12", freq="M")  # last month before the 2022-2023 rate turn
TURN_MODELS = ("naive", "ets", "rate_regression", "lightgbm")
SEGMENT_TITLES = {"full": "Full period", "rate_turn": "Rate turn (target months 2022-2023)"}
COLORS = {
    "naive": "#7f7f7f",
    "drift": "#8c564b",
    "ets": "#1f77b4",
    "sarima": "#17becf",
    "rate_regression": "#d95f02",
    "lightgbm": "#7b3294",
}
LABELS = {
    "naive": "Naive",
    "drift": "Drift",
    "ets": "ETS",
    "sarima": "SARIMA",
    "rate_regression": "Rate regression",
    "lightgbm": "LightGBM",
}


def load_results(results_dir: Path = DEFAULT_RESULTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``(scores, forecasts)`` from the committed result files."""
    scores = pd.read_csv(results_dir / "backtest_scores.csv")
    forecasts = pd.read_csv(results_dir / "backtest_forecasts.csv")
    for column in ("origin", "target"):
        forecasts[column] = pd.PeriodIndex(forecasts[column], freq="M")
    return scores, forecasts


def summary_table(scores: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 6, 12)) -> pd.DataFrame:
    """MAE (EUR bn) and band coverage (%) per model, segment and selected horizon.

    One row per model; columns are ``(segment, metric, horizon)``.
    """
    shown = scores[scores["horizon"].isin(horizons)].copy()
    shown["coverage"] = 100 * shown["coverage"]
    wide = shown.pivot(index="model", columns=["segment", "horizon"], values=["mae", "coverage"])
    wide.columns = wide.columns.reorder_levels([1, 0, 2])
    wide = wide.sort_index(axis=1).loc[list(LABELS)]
    return wide.rename(index=LABELS).rename_axis("Model")


def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.text(0.01, 0.01, SOURCE_NOTE, fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _by_horizon(scores: pd.DataFrame, column: str, ylabel: str, title: str, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=column == "coverage")
    for ax, (segment, subtitle) in zip(axes, SEGMENT_TITLES.items(), strict=True):
        part = scores[scores["segment"] == segment]
        for model, rows in part.groupby("model", sort=False):
            ax.plot(rows["horizon"], rows[column], color=COLORS[model], label=LABELS[model])
        if column == "coverage":
            ax.axhline(part["nominal"].iloc[0], color="black", ls="--", lw=1, label="Nominal")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title(subtitle)
        ax.set_xlabel("Forecast horizon (months)")
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    return _finish(fig, path)


def plot_mae_by_horizon(scores: pd.DataFrame, path: Path) -> Path:
    """MAE (EUR bn) by horizon per model, full period and rate-turn segment."""
    return _by_horizon(
        scores, "mae", "MAE (EUR bn)", "Mean absolute error by forecast horizon", path
    )


def plot_coverage_by_horizon(scores: pd.DataFrame, path: Path) -> Path:
    """Observed share of actuals inside the band by horizon, against the nominal level."""
    return _by_horizon(
        scores, "coverage", "Observed band coverage", "Band coverage by forecast horizon", path
    )


def plot_turn_forecasts(
    forecasts: pd.DataFrame,
    actual: pd.Series,
    path: Path,
    *,
    origin: pd.Period = TURN_ORIGIN,
    models: tuple[str, ...] = TURN_MODELS,
) -> Path:
    """Twelve-month forecasts from ``origin`` against the realised path."""
    part = forecasts[forecasts["origin"] == origin]
    history = actual.loc[origin - 24 : origin + 12]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.index.to_timestamp(), history, color="black", lw=2, label="Realised")
    for model in models:
        rows = part[part["model"] == model]
        x = [origin.to_timestamp(), *rows["target"].dt.to_timestamp()]
        y = [actual.loc[origin], *rows["mean"]]
        ax.plot(x, y, color=COLORS[model], label=LABELS[model])
    ax.axvline(origin.to_timestamp(), color="gray", ls=":")
    ax.set_xlabel("Month")
    ax.set_ylabel("Household overnight deposits, Germany (EUR bn)")
    ax.set_title(f"Forecasts from {origin} (just before the 2022-2023 rate turn)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def make_figures(
    results_dir: Path = DEFAULT_RESULTS_DIR, out_dir: Path = DEFAULT_FIGURE_DIR
) -> list[Path]:
    """Write the three result figures to ``out_dir`` and return their paths."""
    from deposit_forecasting.data import load_panel

    scores, forecasts = load_results(results_dir)
    return [
        plot_mae_by_horizon(scores, out_dir / "mae_by_horizon.png"),
        plot_coverage_by_horizon(scores, out_dir / "coverage_by_horizon.png"),
        plot_turn_forecasts(forecasts, load_panel()[TARGET], out_dir / "rate_turn_forecasts.png"),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="deposit-forecasting-figures",
        description="Draw the backtest result figures from the committed result tables.",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args(argv)
    for path in make_figures(args.results, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
