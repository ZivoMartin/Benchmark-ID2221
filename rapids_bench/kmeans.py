"""10 iterations of GPU Lloyd k-means, preserving CPU Dask initialization."""
import numpy as np
import cupy as cp
import cudf
import dask_cudf


def _clean(part):
    x = part[['fare_amount', 'trip_distance']].astype('float64')
    x = x.dropna()
    x = x[(x['fare_amount'] >= 0) & (x['trip_distance'] >= 0)]
    return x[cp.isfinite(x.to_cupy()).all(axis=1)]


def _partials(part, centers):
    points = cp.asarray(part[['fare_amount', 'trip_distance']].to_cupy(), dtype=cp.float64)
    c = cp.asarray(centers, dtype=cp.float64)
    labels = cp.argmin(cp.sum((points[:, None, :] - c[None, :, :]) ** 2, axis=2), axis=1)
    sums = cp.zeros((len(centers), 2), dtype=cp.float64)
    cp.add.at(sums, labels, points)
    counts = cp.bincount(labels, minlength=len(centers))
    return cudf.DataFrame({'cluster': cp.arange(len(centers), dtype=cp.int32),
                           'sum_fare': sums[:, 0], 'sum_distance': sums[:, 1],
                           'count': counts.astype(cp.int64)})


def run(path, partitions=16, k=5, iterations=10, seed=42):
    if k < 1 or iterations < 1:
        raise ValueError('k and iterations must be positive')
    frame = dask_cudf.read_parquet(path, columns=['fare_amount', 'trip_distance'])
    frame = frame.repartition(npartitions=partitions)
    frame = frame.map_partitions(_clean, meta=cudf.DataFrame({
        'fare_amount': cudf.Series([], dtype='float64'),
        'trip_distance': cudf.Series([], dtype='float64')}))
    # Materialize cleaned points once, like CPU Dask's persist().
    frame = frame.persist()
    try:
        sizes = frame.map_partitions(len, meta=('length', 'int64')).compute()
        n = int(sizes.sum())
        if n < k:
            raise ValueError('Too few valid records')
        positions = np.linspace(0, n - 1, k, dtype=int)
        # Pick initial points by position in the original partition order.
        lengths = np.asarray(sizes.to_numpy(), dtype=int)
        boundaries = np.cumsum(lengths)
        delayed_parts = frame.to_delayed()
        selected = []
        for pos in positions:
            i = int(np.searchsorted(boundaries, pos, side='right'))
            local = int(pos - (boundaries[i-1] if i else 0))
            part = delayed_parts[i].compute()
            selected.append(part.iloc[local][['fare_amount', 'trip_distance']].to_pandas().to_numpy(dtype=np.float64))
        centers = np.stack(selected)
        meta = cudf.DataFrame({'cluster': cudf.Series([], dtype='int32'),
                               'sum_fare': cudf.Series([], dtype='float64'),
                               'sum_distance': cudf.Series([], dtype='float64'),
                               'count': cudf.Series([], dtype='int64')})
        for _ in range(iterations):
            partial = frame.map_partitions(_partials, centers, meta=meta).compute()
            totals = partial.groupby('cluster')[['sum_fare', 'sum_distance', 'count']].sum().to_pandas()
            for j in range(k):
                if j in totals.index and totals.loc[j, 'count'] > 0:
                    centers[j] = totals.loc[j, ['sum_fare', 'sum_distance']].to_numpy(dtype=np.float64) / totals.loc[j, 'count']
        return {'rows': n, 'checksum': float(centers.sum()), 'centers': centers.tolist()}
    finally:
        del frame
