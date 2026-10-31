"""GPU version of the existing two-join CPU Dask workload."""
import cudf
import dask_cudf
import dask


def run(path, zones_path, partitions=16):
    trips = dask_cudf.read_parquet(path, columns=[
        'PULocationID', 'DOLocationID', 'fare_amount', 'trip_distance'
    ]).repartition(npartitions=partitions)
    zones = cudf.read_csv(zones_path)[['LocationID', 'Borough']].rename(
        columns={'LocationID': 'PULocationID'})
    zones['PULocationID'] = zones['PULocationID'].astype(trips._meta['PULocationID'].dtype)
    enriched = trips.merge(zones, on='PULocationID', how='left')
    derived = trips.groupby('DOLocationID').agg({
        'fare_amount': 'mean', 'trip_distance': 'mean'
    }, split_out=partitions).reset_index().rename(columns={
        'fare_amount': 'zone_mean_fare', 'trip_distance': 'zone_mean_distance'
    })
    joined = enriched.merge(derived, on='DOLocationID', how='left')
    # Match existing CPU semantics: the checksum is the original fare sum.
    # Explicitly evaluate zone_mean_fare, as in CPU Dask.
    fare_sum, zone_sum, count = dask.compute(
        joined['fare_amount'].sum(),
        joined['zone_mean_fare'].sum(),
        joined['fare_amount'].count(),
    )
    return {'rows': int(count), 'checksum': float(fare_sum)}
