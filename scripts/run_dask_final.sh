
    
  
#!/usr/bin/env bash
set -Eeuo pipefail

# Run from anywhere; the script lives in the project root after copying it there.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo 'ERROR: .venv/bin/python not found. Create/install the project virtualenv first.' >&2
  exit 1
fi

for size in 100000 500000 1000000 2000000; do
  if [[ ! -e "data/prepared/trips_${size}.parquet" ]]; then
    echo "ERROR: missing data/prepared/trips_${size}.parquet" >&2
    exit 1
  fi
done
if [[ ! -f data/raw/taxi_zone_lookup.csv ]]; then
  echo 'ERROR: missing data/raw/taxi_zone_lookup.csv' >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-results/dask/final_${RUN_ID}}"
mkdir -p "$OUT_DIR/logs"
OUT_DIR="$(realpath "$OUT_DIR")"

cat > "$OUT_DIR/config.txt" <<CONFIG
framework=dask
sizes=100000,500000,1000000,2000000
workers=1,2,4,8
partitions=16
repetitions=3
warmups=1
python=$(.venv/bin/python --version 2>&1)
started_utc=$(date -u +%FT%TZ)
CONFIG
.venv/bin/python -m pip freeze > "$OUT_DIR/requirements.freeze.txt"

failed=0
for workers in 1 2 4 8; do
  for size in 100000 500000 1000000 2000000; do
    for workload in aggregation join kmeans; do
      label="${workload}_${size}_w${workers}"
      csv="$OUT_DIR/${label}.csv"
      log="$OUT_DIR/logs/${label}.log"
      if [[ -f "$csv" ]] && .venv/bin/python - "$csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as f:
    rows = list(csv.DictReader(f))
sys.exit(0 if len(rows) == 3 and all(r['status'] == 'ok' and r['run'] == str(i + 1) for i, r in enumerate(rows)) else 1)
PY
      then
        echo "SKIP completed: $label"
        continue
      fi
      echo "START $(date -u +%FT%TZ) $label"

      if .venv/bin/python -u -m dask_bench.benchmark \
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

.venv/bin/python - "$OUT_DIR" <<'PY'
import csv, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
rows = []
for p in sorted(root.glob('*.csv')):
    if p.name in ('measurements.csv', 'summary.csv'):
        continue
    with p.open(newline='') as f:
        chunk = list(csv.DictReader(f))
    if len(chunk) == 3 and all(r.get('status') == 'ok' and r.get('run') == str(i+1) for i,r in enumerate(chunk)):
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
        w = csv.writer(f)
        w.writerow(['workload','dataset','workers','partitions','runs','median_seconds','min_seconds','max_seconds','median_peak_rss_bytes','median_baseline_rss_bytes','median_delta_rss_bytes','rows','checksum'])
        for (workload,size,workers,parts),rs in sorted(groups.items()):
            w.writerow([workload,size,workers,parts,len(rs),
                statistics.median(float(r['seconds']) for r in rs),
                min(float(r['seconds']) for r in rs),
                max(float(r['seconds']) for r in rs),
                statistics.median(int(r['peak_rss_bytes']) for r in rs),
                statistics.median(int(r['baseline_rss_bytes']) for r in rs),
                statistics.median(int(r['delta_rss_bytes']) for r in rs),
                rs[0]['rows'],rs[0]['checksum']])
print(f'Complete measured runs: {len(rows)}/144')
print(f'Output directory: {root}')
PY

if (( failed > 0 )); then
  echo "$failed configuration(s) failed. Re-run with OUT_DIR='$OUT_DIR' to retry incomplete configurations." >&2
  exit 1
fi

echo 'All benchmark configurations completed.'
