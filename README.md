# Spark, Dask, and RAPIDS Benchmarks

Performance comparison of **Apache Spark**, **Dask**, and **NVIDIA RAPIDS** using NYC Yellow Taxi trip data.

The project evaluates three workloads—aggregation, join, and iterative K-means—on datasets containing **100K, 500K, 1M, and 2M rows**. CPU experiments use 1, 2, 4, and 8 workers; RAPIDS uses a single NVIDIA GPU. Each configuration includes one warmup and three measured executions.

The evaluation covers execution time, scalability, and memory consumption. Results and interpretations are presented in [report.pdf](report.pdf).

## Dataset

The experiments use the [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) from January–March 2024.

Source files:
- [January 2024](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet)
- [February 2024](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet)
- [March 2024](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-03.parquet)

The preparation script downloads the source data and generates four subsets under `data/prepared/`:

```bash
python common/prepare_data.py
```

Prepared datasets are excluded from Git because they can be regenerated.

## Installation

Run all commands from the repository root.

**Dask:**

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Apache Spark:**

```bash
python -m venv .venv-spark
.venv-spark/bin/pip install -r requirements-spark.txt
```

Spark additionally requires Java 17.

**NVIDIA RAPIDS:**

RAPIDS requires a CUDA-compatible GPU and a compatible installation of cuDF and Dask-cuDF. Follow the [official installation instructions](https://docs.rapids.ai/install/).

The GPU experiments were conducted on Google Colab using an NVIDIA Tesla T4. The recorded Python dependencies are available in `results/rapids/final_20261031T181930Z/requirements.freeze.txt`.

## Running the Benchmarks

The runner scripts are located in `scripts/` and should be executed from the repository root.

| Framework | Command |
|---|---|
| Dask | `bash scripts/run_dask_final.sh` |
| Spark | `bash scripts/run_spark_final.sh` |
| RAPIDS | `bash scripts/run_rapids_final.sh` |

Optional smoke tests are available for Spark and RAPIDS:

```bash
bash scripts/run_spark_smoke.sh
bash scripts/run_rapids_smoke.sh
```

**Note:** If the scripts were moved from the repository root into `scripts/`, ensure their project-root detection points to the parent directory of `scripts/`.

## Results and Analysis

The recorded benchmark results are available under:

```text
results/
├── dask/final_20261027T132222Z/
├── spark/final_20261027T134804Z/
└── rapids/final_20261031T181930Z/
```

Each directory contains:

- `measurements.csv` — individual benchmark executions.
- `summary.csv` — aggregated results.
- `requirements.freeze.txt` — recorded dependencies.
- Additional environment metadata where available.

Analysis scripts are located in `analysis/` and `benchmark_evaluation/`. They generate runtime comparisons, CPU scaling curves, and memory visualizations.

The final report is available as [report.pdf](report.pdf), with its LaTeX source and figures in `report/`.

## Reproducibility Notes

The CPU experiments used Python 3.14.8, while RAPIDS ran on Python 3.13 with cuDF 26.06 and a Tesla T4 GPU.

CPU and GPU experiments were conducted on different hardware, so their performance differences should be interpreted as end-to-end comparisons rather than isolated GPU speedups.

The repository includes **324 measured executions** across the three frameworks, allowing the reported results to be inspected without rerunning the benchmarks.
