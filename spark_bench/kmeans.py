import numpy as np
import pyarrow.parquet as pq
from pyspark import StorageLevel
from pyspark.sql import functions as F


def _initial_centers(path, k):
    # Use input-file row order rather than Spark's nondeterministic RDD order.
    frame = pq.read_table(path, columns=['fare_amount', 'trip_distance']).to_pandas()
    points = frame[['fare_amount', 'trip_distance']].to_numpy(dtype=np.float64)
    good = np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1)
    points = points[good]
    if len(points) < k:
        raise ValueError('Too few valid records')
    positions = np.linspace(0, len(points) - 1, k, dtype=int)
    return points[positions].copy(), len(points)


def _partition_sums(iterator, centers):
    k = len(centers)
    sums = np.zeros((k, 2), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for fare, distance in iterator:
        point = np.array((fare, distance), dtype=np.float64)
        cluster = int(np.argmin(np.sum((centers - point) ** 2, axis=1)))
        sums[cluster] += point
        counts[cluster] += 1
    yield (sums, counts)


def run(spark, path, partitions=16, k=5, iterations=10, seed=42):
    if k < 1 or iterations < 1:
        raise ValueError('k and iterations must be positive')
    centers, expected_n = _initial_centers(path, k)
    trips = spark.read.parquet(path).select('fare_amount', 'trip_distance')
    valid = trips.filter(
        F.col('fare_amount').isNotNull() & F.col('trip_distance').isNotNull()
        & (~F.isnan('fare_amount')) & (~F.isnan('trip_distance'))
        & (F.abs('fare_amount') != float('inf'))
        & (F.abs('trip_distance') != float('inf'))
        & (F.col('fare_amount') >= 0) & (F.col('trip_distance') >= 0)
    ).repartition(partitions)
    points = valid.rdd.map(lambda row: (float(row[0]), float(row[1])))
    points.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        n = points.count()
        if n != expected_n:
            raise ValueError(f'Spark/PyArrow filtering mismatch: {n} vs {expected_n}')
        for _ in range(iterations):
            bc = spark.sparkContext.broadcast(centers)
            try:
                partial = points.mapPartitions(lambda it: _partition_sums(it, bc.value)).collect()
            finally:
                bc.unpersist(blocking=True)
            sums = np.zeros((k, 2), dtype=np.float64)
            counts = np.zeros(k, dtype=np.int64)
            for ps, pc in partial:
                sums += ps
                counts += pc
            occupied = counts > 0
            centers[occupied] = sums[occupied] / counts[occupied, None]
        return {'rows': n, 'checksum': float(centers.sum()), 'centers': centers.tolist()}
    finally:
        points.unpersist(blocking=True)
