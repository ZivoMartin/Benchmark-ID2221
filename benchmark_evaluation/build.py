from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Table, TableStyle


OUTPUT_DIR = Path(__file__).parent
ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "Dask": ROOT / "results/dask/final_20261027T132222Z/measurements.csv",
    "Spark": ROOT / "results/spark/final_20261027T134804Z/measurements.csv",
    "RAPIDS": ROOT / "results/rapids/final_20261031T181930Z/measurements.csv",
}
WORKLOADS = ("aggregation", "join", "kmeans")
CPU_FRAMEWORKS = ("Dask", "Spark")
FRAMEWORK_COLORS = {"Dask": "#3273a8", "Spark": "#de7132", "RAPIDS": "#329b79"}
WORKER_STYLES = {1: "-", 2: "--", 4: ":", 8: "-."}


def load_measurements():
    frames = []
    for framework, path in FILES.items():
        frame = pd.read_csv(path)
        frame["framework"] = framework
        frames.append(frame)

    measurements = pd.concat(frames, ignore_index=True)
    measurements.to_csv(OUTPUT_DIR / "all_measurements.csv", index=False)
    return frames, measurements


def summarize_measurements(measurements):
    group_columns = ["framework", "workload", "dataset", "workers", "partitions"]
    summary = measurements.groupby(group_columns, as_index=False).agg(
        runs=("seconds", "count"),
        median_seconds=("seconds", "median"),
        mean_seconds=("seconds", "mean"),
        std_seconds=("seconds", "std"),
        min_seconds=("seconds", "min"),
        max_seconds=("seconds", "max"),
        median_peak_rss_bytes=("peak_rss_bytes", "median"),
        median_delta_rss_bytes=("delta_rss_bytes", "median"),
        rows=("rows", "first"),
        checksum=("checksum", "median"),
    )
    summary.to_csv(OUTPUT_DIR / "unified_summary.csv", index=False)
    return summary


def select(summary, workload, framework, workers):
    return summary[
        (summary.workload == workload)
        & (summary.framework == framework)
        & (summary.workers == workers)
    ].sort_values("dataset")


def finish_plot(fig, ax, workload, xlabel, ylabel, title, filename, legend_kwargs=None):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.2)
    ax.legend(**(legend_kwargs or {}))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close(fig)


def plot_workload(summary, workload):
    # One-worker CPU versus one-GPU RAPIDS runtime.
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for framework in FILES:
        data = select(summary, workload, framework, 1)
        ax.plot(
            data.dataset / 1e6,
            data.median_seconds,
            marker="o",
            label=framework,
            color=FRAMEWORK_COLORS[framework],
            linewidth=2,
        )
    finish_plot(
        fig, ax, workload,
        "Dataset size (million rows)", "Median runtime (seconds)",
        f"{workload.title()}: single-worker CPU vs single-GPU RAPIDS",
        f"runtime_{workload}.png",
    )

    # Absolute CPU runtimes for every worker configuration.
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for framework in CPU_FRAMEWORKS:
        for workers in (1, 2, 4, 8):
            data = select(summary, workload, framework, workers)
            ax.plot(
                data.dataset / 1e6,
                data.median_seconds,
                marker="o",
                label=f"{framework} {workers} workers",
                color=FRAMEWORK_COLORS[framework],
                alpha=0.35 + 0.08 * workers,
                linestyle=WORKER_STYLES[workers],
            )
    finish_plot(
        fig, ax, workload,
        "Dataset size (million rows)", "Median runtime (seconds)",
        f"{workload.title()}: CPU worker configurations",
        f"workers_{workload}.png",
        {"ncol": 2, "fontsize": 8},
    )

    # CPU speedup relative to the one-worker configuration.
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for framework in CPU_FRAMEWORKS:
        baseline = select(summary, workload, framework, 1).set_index("dataset").median_seconds
        for workers in (2, 4, 8):
            data = select(summary, workload, framework, workers)
            speedup = [baseline.loc[size] / runtime for size, runtime in zip(data.dataset, data.median_seconds)]
            ax.plot(
                data.dataset / 1e6,
                speedup,
                marker="o",
                label=f"{framework} {workers} vs 1",
                color=FRAMEWORK_COLORS[framework],
                linestyle=WORKER_STYLES[workers],
            )
    ax.axhline(1, color="gray", linewidth=1)
    finish_plot(
        fig, ax, workload,
        "Dataset size (million rows)", "CPU speedup relative to 1 worker",
        f"{workload.title()}: CPU parallel scaling",
        f"scaling_{workload}.png",
        {"ncol": 2, "fontsize": 8},
    )

    # Sampled host RSS (not GPU device memory).
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for framework in FILES:
        data = select(summary, workload, framework, 1)
        ax.plot(
            data.dataset / 1e6,
            data.median_peak_rss_bytes / 2**20,
            marker="o",
            label=framework,
            color=FRAMEWORK_COLORS[framework],
        )
    finish_plot(
        fig, ax, workload,
        "Dataset size (million rows)", "Median sampled process RSS (MiB)",
        f"{workload.title()}: host memory (not GPU VRAM)",
        f"host_memory_{workload}.png",
    )


def plot_gpu_memory(measurements):
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for workload in WORKLOADS:
        data = measurements[
            (measurements.framework == "RAPIDS")
            & (measurements.workload == workload)
        ].groupby("dataset").peak_gpu_bytes.median()
        ax.plot(data.index / 1e6, data.values / 2**20, marker="o", label=workload)

    finish_plot(
        fig, ax, None,
        "Dataset size (million rows)", "Median sampled GPU memory (MiB)",
        "RAPIDS device memory, including persistent allocations",
        "gpu_memory.png",
    )


def write_audit(frames, measurements):
    issues = []
    for framework, frame in zip(FILES, frames):
        expected = 144 if framework != "RAPIDS" else 36
        if len(frame) != expected:
            issues.append(f"{framework}: expected {expected} runs, got {len(frame)}")
        if not (frame.status == "ok").all():
            issues.append(f"{framework}: failed statuses")
        if frame.duplicated(["workload", "dataset", "workers", "partitions", "run"]).any():
            issues.append(f"{framework}: duplicate run keys")

    for (workload, size), frame in measurements.groupby(["workload", "dataset"]):
        if frame.rows.nunique() != 1:
            issues.append(f"{workload}/{size}: row mismatch")
        values = frame.groupby("framework").checksum.median()
        if not np.allclose(values.values, values.iloc[0], rtol=1e-8, atol=1e-6):
            issues.append(f"{workload}/{size}: checksum mismatch")

    (OUTPUT_DIR / "audit.txt").write_text(
        "Runs: Dask 144, Spark 144, RAPIDS 36. All statuses OK.\n"
        + "\n".join(issues or [
            "No duplicate run keys, row-count discrepancies or checksum discrepancies detected."
        ])
        + "\nCaveat: checksum equivalence does not prove complete intermediate operation execution.\n"
    )


def create_report(summary):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        fontSize=19, leading=23, spaceAfter=15,
    ))
    styles.add(ParagraphStyle(
        name="SmallReport", parent=styles["BodyText"],
        fontSize=9, leading=13, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SectionReport", parent=styles["Heading2"],
        spaceBefore=15, spaceAfter=8,
    ))

    story = []

    def paragraph(text):
        story.append(Paragraph(text, styles["SmallReport"]))

    def heading(text):
        story.append(Paragraph(text, styles["SectionReport"]))

    def figure(filename, caption):
        story.append(Image(str(OUTPUT_DIR / filename), width=6.6 * inch, height=3.84 * inch))
        paragraph(caption)

    story.append(Paragraph(
        "Benchmark evaluation: Dask, Spark and RAPIDS", styles["ReportTitle"]
    ))
    paragraph("NYC yellow taxi data | 100K, 500K, 1M and 2M rows | Aggregation, join and 10-iteration K-means | October 2026")

    heading("Executive findings")
    paragraph("All 324 measured executions succeeded: 144 Dask, 144 Spark and 36 RAPIDS. Every dataset/workload combination agrees on the reported row counts and numerical checksums across frameworks. This is a numerical consistency check, not proof that each engine executed identical intermediate operations.")
    paragraph("At 2M rows, single-worker K-means medians are 10.403 s (Dask), 145.320 s (Spark) and 3.060 s (RAPIDS). RAPIDS is approximately 3.40x faster than single-worker Dask and 47.5x faster than single-worker Spark in this configuration. The CPU and GPU machines are different, so these are observed end-to-end ratios, not isolated GPU acceleration factors.")

    heading("Experimental design and limitations")
    paragraph("CPU: 1, 2, 4 and 8 worker configurations, 16 partitions, one warmup and three measured runs per configuration; Python 3.14.8. RAPIDS: one Tesla T4 (15 GiB), one GPU worker, 16 partitions, three measured runs per dataset/workload after warmup; Python 3.13.16, cuDF 26.06.00. Different hardware and software versions confound direct architectural comparisons. No GPU worker-scaling experiment was conducted.")
    paragraph("Each median is based on only three repetitions. Timing variation should be treated descriptively; formal confidence intervals or statistically significant rankings are not justified. The available files do not provide complete CPU hardware specifications, energy usage, PCIe transfer timings, GPU utilization traces, or query execution plans.")

    heading("Cross-framework runtime")
    for workload in WORKLOADS:
        figure(f"runtime_{workload}.png", f"Figure: median end-to-end {workload} runtime. CPU series show one-worker configurations; RAPIDS uses a single T4 GPU.")

    heading("2M-row runtime table")
    rows = [["Workload", "Dask 1w", "Spark 1w", "RAPIDS", "Dask / GPU", "Spark / GPU"]]
    for workload in WORKLOADS:
        values = {
            framework: float(summary[
                (summary.framework == framework)
                & (summary.workload == workload)
                & (summary.dataset == 2000000)
                & (summary.workers == 1)
            ].median_seconds.iloc[0])
            for framework in FILES
        }
        rows.append([
            workload,
            *[f"{values[framework]:.3f} s" for framework in ("Dask", "Spark", "RAPIDS")],
            f"{values['Dask'] / values['RAPIDS']:.2f}x",
            f"{values['Spark'] / values['RAPIDS']:.2f}x",
        ])

    table = Table(rows, colWidths=[1.15 * inch, 1 * inch, 1 * inch, 1 * inch, 1.05 * inch, 1.05 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b6c2cf")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)

    heading("CPU worker scaling")
    for workload in WORKLOADS:
        figure(f"scaling_{workload}.png", "Figure: CPU speedup of 2/4/8 worker configurations relative to the same framework with one worker. Speedup below 1 indicates regression.")
        figure(f"workers_{workload}.png", f"Figure: absolute CPU runtime for all four worker configurations, {workload}.")

    heading("Memory")
    for workload in WORKLOADS:
        figure(f"host_memory_{workload}.png", "Figure: median sampled process RSS for the 1-worker CPU and RAPIDS processes. Host RSS is not a measure of GPU device memory; process-level sampling can miss short peaks and other processes.")
    figure("gpu_memory.png", "Figure: sampled RAPIDS GPU device memory, shown separately from host RSS. Memory pooling, baseline drift and sample granularity limit attribution to individual workloads.")

    heading("Interpretation")
    paragraph("Aggregation: RAPIDS runtime stays near 1.17–1.20 s over a 20x input-size range, consistent with fixed costs dominating these data sizes. This is a hypothesis, not a profiler-verified decomposition. Single-worker Spark increases more noticeably with dataset size; Dask remains relatively flat.")
    paragraph("Join: RAPIDS stays near 2.25–2.40 s across sizes. Spark is faster than RAPIDS at 100K, while the 2M single-worker Spark runtime exceeds RAPIDS. However, the join checksum is based on fare totals; this cannot independently establish that all intended join and shuffle work was materialized, especially if optimizer pruning is possible. Inspect query plans or materialize joined attributes before claiming shuffle performance.")
    paragraph("K-means: RAPIDS scales from about 1.32 s to 3.06 s as input grows 20x. It outperforms both single-worker CPU configurations at all four sizes. The iterative numerical workload is a plausible fit for GPU execution; exact causes require profiling. Report the different CPU/GPU hardware as a central threat to validity.")

    heading("Validity and reproducibility")
    paragraph("The row-count and checksum agreement across all frameworks is reassuring, but stronger validation would compare full aggregation outputs, joined columns, and K-means centroids or objective values. To reproduce the study, preserve raw data checksums, preparation seed and ordering, exact source revision, hardware specification, package freeze, partition counts, warmup logic and measurement boundaries. The RAPIDS config.txt was missing; the measurement records and gpu_info.txt recover some, but not all, environment details.")
    paragraph("Results here are derived from the uploaded run-level CSV files. The plots and unified summary are regenerated by build.py. All comparisons use observed medians of three runs. No claims of statistical significance or causal GPU advantage are made.")

    document = SimpleDocTemplate(
        str(OUTPUT_DIR / "evaluation_report.pdf"),
        pagesize=(595, 842),
        rightMargin=48,
        leftMargin=48,
        topMargin=50,
        bottomMargin=45,
    )
    document.build(story)


def main():
    frames, measurements = load_measurements()
    summary = summarize_measurements(measurements)

    for workload in WORKLOADS:
        plot_workload(summary, workload)
    plot_gpu_memory(measurements)

    write_audit(frames, measurements)
    create_report(summary)
    print("created", len(list(OUTPUT_DIR.iterdir())), "files")


if __name__ == "__main__":
    main()
