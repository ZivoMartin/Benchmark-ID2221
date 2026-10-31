"""Spark local-mode benchmark: one SparkSession per worker configuration."""
import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from datetime import datetime, timezone

import pyspark
from pyspark.sql import SparkSession
from common.metrics import MemorySampler
from spark_bench import aggregation, join, kmeans

FIELDS = ['framework', 'workload', 'dataset', 'workers', 'partitions', 'run',
          'seconds', 'peak_rss_bytes', 'baseline_rss_bytes', 'delta_rss_bytes',
          'rows', 'checksum', 'python', 'spark_version', 'timestamp_utc', 'status']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workloads', nargs='+', choices=['aggregation', 'join', 'kmeans'],
                   default=['aggregation', 'join', 'kmeans'])
    p.add_argument('--sizes', nargs='+', type=int,
                   default=[100000, 500000, 1000000, 2000000])
    p.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4, 8])
    p.add_argument('--partitions', type=int, default=16)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--output', default='results/spark/measurements.csv')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.partitions < 1 or args.repetitions < 1 or args.warmups < 0 or any(w < 1 for w in args.workers):
        p.error('Invalid counts')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size and not args.overwrite:
        p.error(f'{out} already exists (use --overwrite)')
    with out.open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for workers in args.workers:
        spark = (SparkSession.builder.master(f'local[{workers}]')
                 .appName('DaskSparkTaxiBenchmark')
                 .config('spark.ui.enabled', 'false')
                 .config('spark.sql.shuffle.partitions', str(args.partitions))
                 .config('spark.sql.adaptive.enabled', 'false')
                 .getOrCreate())
        spark.sparkContext.setLogLevel('ERROR')
        try:
            for size in args.sizes:
                path = f'data/prepared/trips_{size}.parquet'
                if not Path(path).exists():
                    raise FileNotFoundError(path)
                for workload in args.workloads:
                    def task():
                        if workload == 'aggregation':
                            return aggregation.run(spark, path, args.partitions)
                        if workload == 'join':
                            return join.run(spark, path, 'data/raw/taxi_zone_lookup.csv', args.partitions)
                        return kmeans.run(spark, path, args.partitions)
                    reference = None
                    for index in range(-args.warmups, args.repetitions):
                        if index < 0:
                            result = task()
                            print(f'warmup {workload} {size} {workers}', flush=True)
                        else:
                            with MemorySampler() as mem:
                                start = time.perf_counter()
                                result = task()
                                seconds = time.perf_counter() - start
                        rows, checksum = int(result['rows']), float(result['checksum'])
                        if not math.isfinite(checksum):
                            raise ValueError('Nonfinite checksum')
                        if reference is None:
                            reference = (rows, checksum)
                        elif rows != reference[0] or not math.isclose(checksum, reference[1], rel_tol=1e-6, abs_tol=1e-6):
                            raise ValueError(f'Inconsistent result: {(rows, checksum)} vs {reference}')
                        if index < 0:
                            continue
                        row = dict(framework='spark', workload=workload, dataset=size,
                                   workers=workers, partitions=args.partitions, run=index+1,
                                   seconds=seconds, peak_rss_bytes=mem.peak,
                                   baseline_rss_bytes=mem.baseline, delta_rss_bytes=mem.delta,
                                   rows=rows, checksum=checksum, python=platform.python_version(),
                                   spark_version=pyspark.__version__,
                                   timestamp_utc=datetime.now(timezone.utc).isoformat(), status='ok')
                        with out.open('a', newline='') as f:
                            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                        print(json.dumps(row), flush=True)
        finally:
            spark.stop()


if __name__ == '__main__':
    main()
