#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="$PROJECT_DIR/.venv-spark/bin/python"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk}"
export PATH="$JAVA_HOME/bin:$PATH"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Spark Python environment not found: $PYTHON" >&2
    exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-results/spark/final_${RUN_ID}}"
mkdir -p "$OUT_DIR/logs"
OUT_DIR="$(realpath "$OUT_DIR")"

if [[ ! -f "$OUT_DIR/config.txt" ]]; then
  cat > "$OUT_DIR/config.txt" <<CONFIG
framework=spark
sizes=100000,500000,1000000,2000000
workers=1,2,4,8
partitions=16
repetitions=3
warmups=1
python=$($PYTHON --version 2>&1)
java=$($JAVA_HOME/bin/java -version 2>&1 | head -n 1)
started_utc=$(date -u +%FT%TZ)
CONFIG
  "$PYTHON" -m pip freeze > "$OUT_DIR/requirements.freeze.txt"
fi

failed=0
for workers in 1 2 4 8; do
  for size in 100000 500000 1000000 2000000; do
    for workload in aggregation join kmeans; do
      label="${workload}_${size}_w${workers}"
      csv="$OUT_DIR/${label}.csv"
      log="$OUT_DIR/logs/${label}.log"

      if [[ -f "$csv" ]] && "$PYTHON" - "$csv" <<'PY'
import csv, sys
try:
    with open(sys.argv[1], newline='') as f:
        rows = list(csv.DictReader(f))
    valid = len(rows) == 3 and all(
        r['status'] == 'ok' and r['run'] == str(i + 1)
        for i, r in enumerate(rows)
    )
except (OSError, KeyError, ValueError):
    valid = False
sys.exit(0 if valid else 1)
PY
      then
        echo "SKIP completed: $label"
        continue
      fi

      echo "START $(date -u +%FT%TZ) $label"
      # Each invocation starts a fresh Python process and Spark JVM.
      # SparkSession startup happens before the runner's timed workload.
      if "$PYTHON" -u -m spark_bench.benchmark \
          --sizes "$size" \
          --workers "$workers" \
          --partitions 16 \
          --workloads "$workload" \
          --repetitions 3 \
          --warmups 1 \
          --output "$csv" \
          --overwrite > "$log" 2>&1; then
        echo "DONE  $(date -u +%FT%TZ) $label"
      else
        code=$?
        echo "FAIL  $(date -u +%FT%TZ) $label (exit $code; see $log)" >&2
        failed=$((failed + 1))
      fi
    done
  done
done

"$PYTHON" - "$OUT_DIR" <<'PY'
import csv
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for p in sorted(root.glob('*.csv')):
    if p.name in ('measurements.csv', 'summary.csv'):
        continue
    with p.open(newline='') as f:
        chunk = list(csv.DictReader(f))
    if len(chunk) == 3 and all(
        r.get('status') == 'ok' and r.get('run') == str(i + 1)
        for i, r in enumerate(chunk)
    ):
        rows.extend(chunk)

if rows:
    with (root / 'measurements.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for r in rows:
        key = (r['workload'], int(r['dataset']), int(r['workers']), int(r['partitions']))
        groups.setdefault(key, []).append(r)

    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'workload', 'dataset', 'workers', 'partitions', 'runs',
            'median_seconds', 'min_seconds', 'max_seconds',
            'median_peak_rss_bytes', 'median_baseline_rss_bytes',
            'median_delta_rss_bytes', 'rows', 'checksum'
        ])
        for (workload, size, workers, partitions), group in sorted(groups.items()):
            writer.writerow([
                workload, size, workers, partitions, len(group),
                statistics.median(float(r['seconds']) for r in group),
                min(float(r['seconds']) for r in group),
                max(float(r['seconds']) for r in group),
                statistics.median(int(r['peak_rss_bytes']) for r in group),
                statistics.median(int(r['baseline_rss_bytes']) for r in group),
                statistics.median(int(r['delta_rss_bytes']) for r in group),
                group[0]['rows'], group[0]['checksum']
            ])

print(f'Complete measured runs: {len(rows)}/144')
print(f'Output directory: {root}')
if len(rows) != 144:
    sys.exit('ERROR: Benchmark incomplete; resume using the same OUT_DIR.')
PY

if (( failed > 0 )); then
  echo "$failed configuration(s) failed; resume with OUT_DIR='$OUT_DIR'" >&2
  exit 1
fi

