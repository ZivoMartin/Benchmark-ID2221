"""GPU version of the CPU Dask pickup-zone/hour aggregation."""
import dask_cudf


def run(path, partitions=16):
    df = dask_cudf.read_parquet(path, columns=[
        'PULocationID', 'tpep_pickup_datetime', 'fare_amount', 'trip_distance'
    ]).repartition(npartitions=partitions)
    df = df.assign(hour=df.tpep_pickup_datetime.dt.hour)
    # Compute all outputs, including mean columns (not just the checksum).
    result = df.groupby(['PULocationID', 'hour']).agg({
        'fare_amount': ['count', 'mean'], 'trip_distance': 'mean'
    }, split_out=partitions).compute()
    return {'rows': len(result), 'checksum': float(result[('fare_amount', 'count')].sum())}
