"""Command-line interface: ``solarcast evaluate``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from solarcast.data import load_dataset
from solarcast.evaluation import HORIZONS, run_protocol, score
from solarcast.features import CLEAR, CLOUDY, build_dataset

ALL_MODELS = ("lightgbm", "lstm", "chronos")
DEFAULT_MODELS = ("lightgbm", "lstm")


def _make_models(names: Sequence[str]) -> list:
    """Instantiate learned models by name, skipping unavailable optional ones."""
    from solarcast import models

    available = {
        "lightgbm": models.LightGBMModel,
        "lstm": models.LSTMModel,
        "chronos": models.ChronosModel,
    }
    return [available[n]() for n in names]


def format_table(metrics: pd.DataFrame, protocol: str) -> str:
    """Render the model x horizon x regime table as text.

    Args:
        metrics: Output of :func:`solarcast.evaluation.score`.
        protocol: Which protocol to show.

    Returns:
        Multi-line string.
    """
    lines = []
    sub = metrics[metrics["protocol"] == protocol]
    for regime in (CLEAR, CLOUDY, "all"):
        s = sub[sub["regime"] == regime]
        if s.empty:
            continue
        n = int(s.groupby("horizon_min")["n"].max().iloc[0])
        lines.append(f"\n[{protocol}] {regime} days (daylight steps per horizon ~ {n})")
        lines.append("RMSE kWh/15min (skill vs persistence)")
        piv = {}
        for (model, hz), g in s.groupby(["model", "horizon_min"]):
            r = g.iloc[0]
            piv.setdefault(model, {})[hz] = f"{r['rmse']:.3f} ({r['skill_vs_persistence']:+.2f})"
        table = pd.DataFrame(piv).T
        table.columns = [f"{c} min" for c in table.columns]
        lines.append(table.to_string())
    return "\n".join(lines)


def plot_rmse(metrics: pd.DataFrame, path: Path, protocol: str = "holdout") -> None:
    """Save RMSE-versus-horizon curves for clear and cloudy days.

    Args:
        metrics: Output of :func:`solarcast.evaluation.score`.
        path: Output image path.
        protocol: Protocol to plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, regime in zip(axes, (CLEAR, CLOUDY), strict=True):
        s = metrics[(metrics["protocol"] == protocol) & (metrics["regime"] == regime)]
        for model, g in s.groupby("model"):
            g = g.sort_values("horizon_min")
            ax.plot(g["horizon_min"], g["rmse"], marker="o", label=model)
        ax.set_title(f"{regime} days ({protocol})")
        ax.set_xlabel("horizon (min)")
        ax.set_xticks([15, 30, 45, 60])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMSE (kWh / 15 min, daylight)")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> int:
    """Run the ``evaluate`` sub-command.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    data_dir = Path(args.data_dir)
    frame = load_dataset(data_dir / "Generation data.xlsx", data_dir / "Irradiation data.xlsx")
    ds = build_dataset(frame)
    labels = ds.day_label.groupby(ds.day_label.index.normalize()).first().value_counts()
    print(f"Loaded {len(frame)} rows {frame.index.min()} -> {frame.index.max()}")
    print(f"Day labels: {labels.to_dict()}")
    learned = _make_models(args.models)
    protocols = ["holdout", "rolling"] if args.protocol == "both" else [args.protocol]
    parts = [score(run_protocol(ds, p, learned, HORIZONS), ds) for p in protocols]
    metrics = pd.concat(parts, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    for p in protocols:
        print(format_table(metrics, p))
    plot_rmse(metrics, out_dir / "rmse_by_horizon.png", protocols[0])
    print(f"\nSaved {out_dir / 'metrics.csv'} and {out_dir / 'rmse_by_horizon.png'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``solarcast`` command.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="solarcast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="Evaluate baselines and models")
    ev.add_argument("--data-dir", default=".", help="Folder holding the two .xlsx files")
    ev.add_argument("--out-dir", default="results")
    ev.add_argument("--protocol", choices=["holdout", "rolling", "both"], default="both")
    ev.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        choices=ALL_MODELS,
        help="Learned models to include (baselines always run)",
    )
    ev.set_defaults(func=evaluate)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
