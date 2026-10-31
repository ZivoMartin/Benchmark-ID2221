"""Match the Dask pickup-zone/hour aggregation."""
from pyspark.sql import functions as F


def run(spark, path, partitions=16):
    trips = spark.read.parquet(path).select(
        'PULocationID', 'tpep_pickup_datetime', 'fare_amount', 'trip_distance'
    ).repartition(partitions)
    grouped = (trips.withColumn('hour', F.hour('tpep_pickup_datetime'))
               .groupBy('PULocationID', 'hour')
               .agg(F.count('fare_amount').alias('fare_count'),
                    F.avg('fare_amount').alias('mean_fare'),
                    F.avg('trip_distance').alias('mean_distance')))
    # Materialize every aggregated column, as Dask's .compute() does.
    rows = grouped.orderBy('PULocationID', 'hour').collect()
    return {'rows': len(rows), 'checksum': float(sum(r['fare_count'] for r in rows))}
