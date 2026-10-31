"""Single-T4 GPU benchmark. --partitions is task granularity, NOT GPU count."""
import argparse
import csv
import json
import math
import platform
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import cudf
import dask
import dask_cudf
from common.metrics import MemorySampler
from rapids_bench import aggregation, join, kmeans

FIELDS = ['framework', 'workload', 'dataset', 'workers', 'partitions', 'run',
          'seconds', 'peak_rss_bytes', 'baseline_rss_bytes', 'delta_rss_bytes',
          'peak_gpu_bytes', 'baseline_gpu_bytes', 'delta_gpu_bytes',
          'rows', 'checksum', 'python', 'cudf_version', 'dask_version',
          'gpu_name', 'timestamp_utc', 'status']


def gpu_used():
    # CUDA driver free/total: process-independent global GPU memory usage.
    free, total = cp.cuda.runtime.memGetInfo()
    return int(total - free)


class GPUMemorySampler:
    def __init__(self):
        self.stop = threading.Event()
        self.baseline = 0
        self.peak = 0

    def __enter__(self):
        self.baseline = self.peak = gpu_used()
        self.stop.clear()
        def poll():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, gpu_used())
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, gpu_used())


def run_task(workload, path, partitions):
    if workload == 'aggregation':
        return aggregation.run(path, partitions)
    if workload == 'join':
        return join.run(path, 'data/raw/taxi_zone_lookup.csv', partitions)
    return kmeans.run(path, partitions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workloads', nargs='+', choices=['aggregation', 'join', 'kmeans'],
                   default=['aggregation', 'join', 'kmeans'])
    p.add_argument('--sizes', nargs='+', type=int, default=[100000, 500000, 1000000, 2000000])
    p.add_argument('--partitions', nargs='+', type=int, default=[16])
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--output', default='results/rapids/measurements.csv')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if any(x < 1 for x in args.sizes + args.partitions) or args.repetitions < 1 or args.warmups < 0:
        p.error('Invalid arguments')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size and not args.overwrite:
        p.error('Output already exists; use --overwrite')
    gpu = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    with output.open('w', newline='') as f:
        csv.DictWriter(f, FIELDS).writeheader()
    # Synchronous Dask execution: one GPU, no misleading multi-worker comparison.
    with dask.config.set(scheduler='synchronous'):
        for partitions in args.partitions:
            for size in args.sizes:
                path = f'data/prepared/trips_{size}.parquet'
                if not Path(path).exists():
                    raise FileNotFoundError(path)
                for workload in args.workloads:
                    reference = None
                    for index in range(-args.warmups, args.repetitions):
                        cp.cuda.Stream.null.synchronize()
                        if index >= 0:
                            with MemorySampler() as mem, GPUMemorySampler() as gmem:
                                t0 = time.perf_counter()
                                result = run_task(workload, path, partitions)
                                cp.cuda.Stream.null.synchronize()
                                seconds = time.perf_counter() - t0

                        else:
                            result = run_task(workload, path, partitions)
                            cp.cuda.Stream.null.synchronize()
                        rows, checksum = int(result['rows']), float(result['checksum'])
                        if not math.isfinite(checksum):
                            raise ValueError('Non-finite checksum')
                        if reference is None:
                            reference = (rows, checksum)
                        elif rows != reference[0] or not math.isclose(checksum, reference[1], rel_tol=1e-6, abs_tol=1e-6):
                            raise ValueError(f'Inconsistent {workload} output: {(rows, checksum)} vs {reference}')
                        if index < 0:
                            print(f'WARMUP {workload} size={size} partitions={partitions}', flush=True)
                            continue
                        row = dict(framework='rapids', workload=workload, dataset=size,
                                   workers=1, partitions=partitions, run=index+1,
                                   seconds=seconds, peak_rss_bytes=mem.peak,
                                   baseline_rss_bytes=mem.baseline, delta_rss_bytes=mem.delta,
                                   peak_gpu_bytes=gmem.peak, baseline_gpu_bytes=gmem.baseline,
                                   delta_gpu_bytes=max(0, gmem.peak-gmem.baseline),
                                   rows=rows, checksum=checksum, python=platform.python_version(),
                                   cudf_version=cudf.__version__, dask_version=dask.__version__,
                                   gpu_name=gpu, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                   status='ok')
                        with output.open('a', newline='') as f:
                            csv.DictWriter(f, FIELDS).writerow(row)
                        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
