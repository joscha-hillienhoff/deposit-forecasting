"""SHAP explanations of the gradient-boosting forecaster.

Fits the model on the full history and explains the median model of selected horizons with
``shap.TreeExplainer``. Values are in log-change units of the cumulative deposit change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from deposit_forecasting.data import load_panel  # noqa: E402
from deposit_forecasting.gbm import GradientBoostingForecaster  # noqa: E402

EXPLAIN_HORIZONS = (1, 12)
DEFAULT_FIGURE_DIR = Path("docs/figures")


def shap_values(model: GradientBoostingForecaster, horizon: int) -> shap.Explanation:
    """SHAP values of the median model of ``horizon`` on its own training rows."""
    X = model.train_X_[horizon]
    return shap.TreeExplainer(model.models_[horizon]["mean"])(X)


def mean_abs_shap(model: GradientBoostingForecaster, horizons=EXPLAIN_HORIZONS) -> pd.DataFrame:
    """Tidy table (``horizon``, ``feature``, ``mean_abs_shap``), largest first per horizon."""
    frames = []
    for h in horizons:
        values = shap_values(model, h)
        frames.append(
            pd.DataFrame(
                {
                    "horizon": h,
                    "feature": values.feature_names,
                    "mean_abs_shap": abs(values.values).mean(axis=0),
                }
            ).sort_values("mean_abs_shap", ascending=False)
        )
    return pd.concat(frames, ignore_index=True)


def save_summary_plots(
    model: GradientBoostingForecaster, out_dir: Path, horizons=EXPLAIN_HORIZONS
) -> list[Path]:
    """Write ``shap_h<h>.png`` beeswarm summary plots and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for h in horizons:
        values = shap_values(model, h)
        shap.summary_plot(values.values, model.train_X_[h], show=False)
        plt.title(f"SHAP values, {h}-month horizon (cumulative log change)")
        path = out_dir / f"shap_h{h}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="deposit-forecasting-explain",
        description="Fit the gradient-boosting model on the full history and explain it with SHAP.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="figure directory (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    model = GradientBoostingForecaster().fit(load_panel())
    print(mean_abs_shap(model).round(4).to_string(index=False))
    for path in save_summary_plots(model, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
