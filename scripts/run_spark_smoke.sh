#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON="$(pwd)/.venv-spark/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
mkdir -p results/spark
"$PYSPARK_PYTHON" -m spark_bench.benchmark \
  --sizes 100000 --workers 1 2 --partitions 16 \
  --warmups 1 --repetitions 3 \
  --output results/spark/smoke_test.csv --overwrite \
  2>&1 | tee results/spark/smoke_test.log
