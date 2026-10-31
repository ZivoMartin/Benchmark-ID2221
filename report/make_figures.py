
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
SUMMARY_FILE = ROOT / "benchmark_evaluation" / "unified_summary.csv"

WORKLOADS = ("aggregation", "join", "kmeans")
CPU_FRAMEWORKS = ("Dask", "Spark")
WORKER_COUNTS = (1, 2, 4, 8)
DATASET_SIZE = 2_000_000

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
})

RUNTIME_SERIES = (
    ("Dask", 1, "-o"),
    ("Spark", 1, "-s"),
    ("RAPIDS", 1, "-^"),
    ("Dask", 4, "--o"),
    ("Spark", 4, "--s"),
)


def select_results(data, workload, framework, workers=None, dataset=None):
    """Select benchmark results for a given configuration."""
    mask = (
        (data["workload"] == workload)
        & (data["framework"] == framework)
    )

    if workers is not None:
        mask &= data["workers"] == workers

    if dataset is not None:
        mask &= data["dataset"] == dataset

    return data.loc[mask]


def add_legend(fig, ax, y_anchor=-0.12):
    """Add a shared legend below the three subplots."""
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        bbox_to_anchor=(0.5, y_anchor),
        frameon=False,
    )


def configure_axis(ax, workload, xlabel):
    """Apply common axis formatting."""
    ax.set_title(workload.capitalize())
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25)


def plot_runtime(data):
    """Plot median runtime against dataset size."""
    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.1, 2.05),
        layout="constrained",
    )

    for ax, workload in zip(axes, WORKLOADS):
        for framework, workers, style in RUNTIME_SERIES:
            results = select_results(
                data, workload, framework, workers=workers
            ).sort_values("dataset")

            label = (
                f"{framework} (4 CPU workers)"
                if workers == 4
                else f"{framework} (1 worker)"
            )

            ax.plot(
                results["dataset"] / 1e6,
                results["median_seconds"],
                style,
                markersize=3,
                linewidth=1.2,
                label=label,
            )

        configure_axis(ax, workload, "Rows (millions)")
        ax.set_yscale("log" if workload == "kmeans" else "linear")

    axes[0].set_ylabel("Median time (s)")
    add_legend(fig, axes[0])

    fig.savefig(REPORT_DIR / "runtime.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_scaling(data):
    """Plot CPU speedup relative to one worker at 2M rows."""
    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.1, 1.9),
        layout="constrained",
    )

    for ax, workload in zip(axes, WORKLOADS):
        for framework in CPU_FRAMEWORKS:
            results = select_results(
                data,
                workload,
                framework,
                dataset=DATASET_SIZE,
            ).sort_values("workers")

            baseline = results.loc[
                results["workers"] == 1, "median_seconds"
            ].iloc[0]

            speedup = baseline / results["median_seconds"]

            ax.plot(
                results["workers"],
                speedup,
                "-o",
                markersize=3,
                label=framework,
            )

        ax.plot(
            [1, 8],
            [1, 8],
            ":",
            color="gray",
            label="Ideal",
        )

        configure_axis(ax, workload, "CPU workers")
        ax.set_xticks(WORKER_COUNTS)

    axes[0].set_ylabel("Speedup vs 1 worker")
    add_legend(fig, axes[0], y_anchor=-0.13)

    fig.savefig(REPORT_DIR / "scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    data = pd.read_csv(SUMMARY_FILE)

    plot_runtime(data)
    plot_scaling(data)


if __name__ == "__main__":
    main()
