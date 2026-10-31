"""Deterministic Lloyd k-means on fare and distance; no framework ML shortcuts."""

import numpy as np
import dask.dataframe as dd
import dask.array as da


def run(path, partitions, k=5, iterations=10, seed=42):
    if k < 1 or iterations < 1:
        raise ValueError("k and iterations must be positive")

    frame = dd.read_parquet(
        path,
        columns=["fare_amount", "trip_distance"],
    ).repartition(npartitions=partitions)


    frame = frame.dropna()

    frame = frame.map_partitions(
        lambda part: part[
            np.isfinite(part["fare_amount"])
            & np.isfinite(part["trip_distance"])
            & (part["fare_amount"] >= 0)
            & (part["trip_distance"] >= 0)
        ],
        meta=frame._meta,
    )

    arr = frame[
        ["fare_amount", "trip_distance"]
    ].to_dask_array(lengths=True).astype("float64")

    arr = arr.persist()

    try:
        n = int(arr.shape[0])

        if n < k:
            raise ValueError("Too few valid records")

        positions = np.linspace(0, n - 1, k, dtype=int)

        centers = np.asarray(
            arr[positions].compute()
        )

        for _ in range(iterations):
            distances = (
                (arr[:, None, :] - centers[None, :, :]) ** 2
            ).sum(axis=2)

            labels = distances.argmin(axis=1)

            sums = da.stack([
                da.where(
                    labels[:, None] == j,
                    arr,
                    0,
                ).sum(axis=0)
                for j in range(k)
            ])

            counts = da.stack([
                (labels == j).sum()
                for j in range(k)
            ])

            sums_v, counts_v = da.compute(
                sums,
                counts,
            )

            centers = np.where(
                counts_v[:, None] > 0,
                sums_v / np.maximum(counts_v[:, None], 1),
                centers,
            )

        return {
            "rows": n,
            "checksum": float(centers.sum()),
            "centers": centers.tolist(),
        }

    finally:
        del arr
