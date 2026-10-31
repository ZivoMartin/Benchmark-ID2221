#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-results/rapids/final_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
# Resolve output path before using it in subprocesses.
OUT="$(realpath "$OUT")"
cat > "$OUT/config.txt" <<CONFIG
framework=rapids
sizes=100000,500000,1000000,2000000
workers=1
partitions=16
repetitions=3
warmups=1
scheduler=synchronous
timestamp_utc=$(date -u +%FT%TZ)
CONFIG
python -m pip freeze > "$OUT/requirements.freeze.txt"
nvidia-smi > "$OUT/gpu_info.txt"
python -m rapids_bench.benchmark \
  --sizes 100000 500000 1000000 2000000 \
  --partitions 16 --repetitions 3 --warmups 1 \
  --output "$OUT/measurements.csv"
python common/validate.py "$OUT/measurements.csv" --repetitions 3
python -m rapids_bench.summarize "$OUT/measurements.csv" --repetitions 3
printf 'Results saved to %s\n' "$OUT"
