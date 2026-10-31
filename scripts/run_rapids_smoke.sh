#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m rapids_bench.benchmark --sizes 100000 --partitions 16 --repetitions 1 --warmups 0 --output results/rapids/smoke_test.csv --overwrite
python common/validate.py results/rapids/smoke_test.csv --repetitions 1
