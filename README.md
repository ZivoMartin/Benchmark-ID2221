# Dask and Spark Benchmarks

Benchmarks comparing **Dask** and **Apache Spark** on NYC Yellow Taxi data.

Each framework runs three workloads: **aggregation**, **join**, and **iterative K-means**. The experiments use datasets of **100K, 500K, 1M, and 2M rows**, with **1, 2, 4, and 8 workers**. Each configuration includes one warm-up and three measured runs.

## Setup

Run all commands from the repository root. Use separate Python environments for Dask and Spark:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

python -m venv .venv-spark
.venv-spark/bin/pip install -r requirements-spark.txt
```

Spark also requires a working Java installation (Java 17 was used for these experiments). The Spark runner uses `.venv-spark/bin/python`.

## Data

The benchmarks read prepared Parquet files from `data/prepared/`. These files are not committed to Git.

Download the NYC TLC Yellow Taxi Parquet source files into `data/raw/`, then use the preparation utility in `common/prepare_data.py` to create the four datasets. The taxi-zone lookup CSVs are stored under `data/`.

> Check the preparation script's command-line interface before running it; its exact arguments are not documented here.

## Run the experiments

**Dask:**

```bash
./run_dask_final.sh
```

**Spark:**

```bash
./run_spark_final.sh
```

**Spark smoke test (optional):**

```bash
./run_spark_smoke.sh
```

If a script is not executable, run `chmod +x run_*.sh` first.

The full runs may take some time. Run each framework using its corresponding environment as configured in the scripts.

## Results

The final results are included in the repository:

- `results/dask/final_20261027T132222Z/`
- `results/spark/final_20261027T134804Z/`

Each directory contains:

- `measurements.csv` — individual timed runs
- `summary.csv` — aggregated results
- `config.txt` — experiment configuration
- `requirements.freeze.txt` — recorded Python dependencies

Per-configuration CSVs and execution logs may be generated locally but are not needed to use the final summaries.

## Plots

Plotting scripts are in `analysis/`:

```bash
python analysis/plot_runtime.py
python analysis/plot_memory.py
python analysis/plot_scaling.py
python analysis/plot_comparison.py
```

These scripts may require input/output paths or other arguments; consult their command-line options before running them.

## Notes

The reported results were collected on a specific machine and are sensitive to hardware, software versions, and execution settings. When comparing frameworks, use the same input data and document differences in execution environments.
# Benchmark-ID2221
