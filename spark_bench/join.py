"""Match the existing Dask join workload and checksum semantics."""
from pyspark.sql import functions as F


def run(spark, path, zones_path, partitions=16):
    trips = spark.read.parquet(path).select(
        'PULocationID', 'DOLocationID', 'fare_amount', 'trip_distance'
    ).repartition(partitions)
    zones = (spark.read.option('header', True).option('inferSchema', True)
             .csv(zones_path).select(F.col('LocationID').cast('int').alias('PULocationID'),
                                     'Borough'))
    enriched = trips.join(F.broadcast(zones), on='PULocationID', how='left')
    derived = (trips.groupBy('DOLocationID')
               .agg(F.avg('fare_amount').alias('zone_mean_fare'),
                    F.avg('trip_distance').alias('zone_mean_distance')))
    joined = enriched.join(derived, on='DOLocationID', how='left')
    # This reproduces the CURRENT Dask benchmark: it touches zone_mean_fare
    # but does not force Borough or zone_mean_distance to be materialized.
    row = joined.agg(F.sum('fare_amount').alias('fare_sum'),
                     F.sum('zone_mean_fare').alias('zone_sum'),
                     F.count('fare_amount').alias('row_count')).first()
    return {'rows': int(row['row_count']), 'checksum': float(row['fare_sum'])}
