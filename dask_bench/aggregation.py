"""Aggregation workload: pickup zone and pickup hour."""
import dask.dataframe as dd

def build(path, partitions):
    df = dd.read_parquet(path, columns=['PULocationID','tpep_pickup_datetime','fare_amount','trip_distance'])
    df = df.repartition(npartitions=partitions)
    df = df.assign(hour=df.tpep_pickup_datetime.dt.hour)
    grouped = df.groupby(['PULocationID','hour'])
    return grouped.agg({'fare_amount':['count','mean'], 'trip_distance':'mean'}, split_out=partitions)

def run(path, partitions):
    result = build(path, partitions).compute().sort_index()
    return {'rows':len(result), 'checksum':float(result[('fare_amount','count')].sum())}
