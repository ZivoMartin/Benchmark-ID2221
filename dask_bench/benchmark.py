"""Run reproducible Dask benchmarks and write CSV measurements."""

import argparse
import csv
import json
import math
import platform
import time

from datetime import datetime, timezone
from pathlib import Path

import dask
import distributed
import pandas as pd
import pyarrow

from distributed import Client, LocalCluster

from common.metrics import MemorySampler
from dask_bench import aggregation, join, kmeans


FIELDS = [
    "framework",
    "workload",
    "dataset",
    "workers",
    "partitions",
    "run",
    "seconds",
    "peak_rss_bytes",
    "baseline_rss_bytes",
    "delta_rss_bytes",
    "rows",
    "checksum",
    "python",
    "dask_version",
    "distributed_version",
    "pandas_version",
    "pyarrow_version",
    "timestamp_utc",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["aggregation", "join", "kmeans"],
        default=["aggregation", "join", "kmeans"],
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100_000, 500_000, 1_000_000, 2_000_000],
    )

    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--partitions",
        type=int,
        default=16,
        help="Fixed partition count across all worker configurations.",
    )

    parser.add_argument(
        "--partitions-per-worker",
        type=int,
        default=None,
        help="Optional alternative: partitions = workers * value.",
    )

    parser.add_argument(
        "--output",
        default="results/dask/measurements.csv",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of appending.",
    )

    return parser.parse_args()


def run_workload(workload, path, partitions):
    if workload == "aggregation":
        return aggregation.run(str(path), partitions)

    if workload == "join":
        return join.run(
            str(path),
            "data/raw/taxi_zone_lookup.csv",
            partitions,
        )

    if workload == "kmeans":
        return kmeans.run(str(path), partitions)

    raise ValueError(f"Unknown workload: {workload}")


def validate_result(result, reference):
    rows = int(result["rows"])
    checksum = float(result["checksum"])

    if rows < 0:
        raise ValueError("Negative row count")

    if not math.isfinite(checksum):
        raise ValueError("Non-finite checksum")

    if reference is not None:
        if rows != reference["rows"]:
            raise ValueError(
                f"Row mismatch: {rows} != {reference['rows']}"
            )

        if not math.isclose(
            checksum,
            reference["checksum"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"Checksum mismatch: {checksum} != "
                f"{reference['checksum']}"
            )

    return {
        "rows": rows,
        "checksum": checksum,
    }


def main():
    args = parse_args()

    if any(w < 1 for w in args.workers):
        raise ValueError("Worker counts must be positive")

    if any(s < 1 for s in args.sizes):
        raise ValueError("Dataset sizes must be positive")

    if args.partitions < 1:
        raise ValueError("Partition count must be positive")

    if args.partitions_per_worker is not None:
        if args.partitions_per_worker < 1:
            raise ValueError("Partitions per worker must be positive")

    if args.repetitions < 1 or args.warmups < 0:
        raise ValueError("Invalid repetition/warmup count")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and output.stat().st_size > 0:
        if not args.overwrite:
            raise FileExistsError(
                f"{output} already exists. "
                "Use --overwrite or choose another --output."
            )

        output.unlink()

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

    for workers in args.workers:
        partitions = (
            workers * args.partitions_per_worker
            if args.partitions_per_worker is not None
            else args.partitions
        )

        with LocalCluster(
            n_workers=workers,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
        ) as cluster:

            with Client(cluster) as client:
                print(
                    f"Workers: {workers}; "
                    f"partitions: {partitions}; "
                    f"dashboard: {client.dashboard_link}",
                    flush=True,
                )

                for size in args.sizes:
                    path = Path(
                        f"data/prepared/trips_{size}.parquet"
                    )

                    if not path.exists():
                        raise FileNotFoundError(path)

                    for workload in args.workloads:
                        reference = None

                        for index in range(
                            -args.warmups,
                            args.repetitions,
                        ):
                            if index < 0:
                                result = run_workload(
                                    workload,
                                    path,
                                    partitions,
                                )

                                reference = validate_result(
                                    result,
                                    reference,
                                )

                                print(
                                    f"warmup {workload} "
                                    f"{size} {workers}",
                                    flush=True,
                                )

                                continue

                            with MemorySampler() as mem:
                                start = time.perf_counter()

                                result = run_workload(
                                    workload,
                                    path,
                                    partitions,
                                )

                                seconds = (
                                    time.perf_counter() - start
                                )

                            reference = validate_result(
                                result,
                                reference,
                            )

                            row = {
                                "framework": "dask",
                                "workload": workload,
                                "dataset": size,
                                "workers": workers,
                                "partitions": partitions,
                                "run": index + 1,
                                "seconds": seconds,
                                "peak_rss_bytes": mem.peak,
                                "baseline_rss_bytes": mem.baseline,
                                "delta_rss_bytes": mem.delta,
                                "rows": reference["rows"],
                                "checksum": reference["checksum"],
                                "python": platform.python_version(),
                                "dask_version": dask.__version__,
                                "distributed_version": distributed.__version__,
                                "pandas_version": pd.__version__,
                                "pyarrow_version": pyarrow.__version__,
                                "timestamp_utc": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                                "status": "ok",
                            }

                            with output.open("a", newline="") as file:
                                writer = csv.DictWriter(
                                    file,
                                    fieldnames=FIELDS,
                                )
                                writer.writerow(row)

                            print(
                                json.dumps(row),
                                flush=True,
                            )


if __name__ == "__main__":
    main()
