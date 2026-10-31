"""Create a CPU-compatible RAPIDS summary from measurements.csv.

Usage: python -m rapids_bench.summarize results/rapids/final_.../measurements.csv
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(source: Path, destination: Path, repetitions: int = 3) -> None:
    with source.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'No measurements in {source}')
    groups = defaultdict(list)
    for row in rows:
        if row['status'] != 'ok':
            raise ValueError(f'Unsuccessful measurement: {row}')
        key = (row['workload'], int(row['dataset']), int(row['workers']), int(row['partitions']))
        groups[key].append(row)
    columns = [
        'workload', 'dataset', 'workers', 'partitions', 'runs',
        'median_seconds', 'min_seconds', 'max_seconds',
        'median_peak_rss_bytes', 'median_baseline_rss_bytes', 'median_delta_rss_bytes',
        'rows', 'checksum',
        'median_peak_gpu_bytes', 'median_baseline_gpu_bytes', 'median_delta_gpu_bytes',
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for key, group in sorted(groups.items()):
            runs = sorted(int(row['run']) for row in group)
            if runs != list(range(1, repetitions + 1)):
                raise ValueError(f'Incomplete/duplicate runs for {key}: {runs}')
            if len({int(row['rows']) for row in group}) != 1:
                raise ValueError(f'Inconsistent row count for {key}')
            checksums = [float(row['checksum']) for row in group]
            if max(checksums) - min(checksums) > 1e-6 * max(1.0, abs(statistics.mean(checksums))):
                raise ValueError(f'Inconsistent checksum for {key}')
            median = lambda col: statistics.median(float(row[col]) for row in group)
            writer.writerow([
                *key, len(group), median('seconds'),
                min(float(row['seconds']) for row in group),
                max(float(row['seconds']) for row in group),
                median('peak_rss_bytes'), median('baseline_rss_bytes'), median('delta_rss_bytes'),
                group[0]['rows'], group[0]['checksum'],
                median('peak_gpu_bytes'), median('baseline_gpu_bytes'), median('delta_gpu_bytes'),
            ])
    print(f'Created {destination}: {len(groups)} configurations, {len(rows)} measurements')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    summarize(args.measurements, args.output or args.measurements.with_name('summary.csv'), args.repetitions)


if __name__ == '__main__':
    main()
