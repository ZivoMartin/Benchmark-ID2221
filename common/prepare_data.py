"""Download TLC parquet and produce deterministic size-controlled subsets."""
import argparse
from pathlib import Path
import requests
import pyarrow as pa
import pyarrow.parquet as pq

URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet'
ZONES = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv'
COLS = ['tpep_pickup_datetime', 'PULocationID', 'DOLocationID', 'fare_amount', 'trip_distance']

def download(url, path):
    if path.exists():
        print('Existing:', path); return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk: f.write(chunk)
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', nargs='+', default=['2024-01', '2024-02', '2024-03'])
    ap.add_argument('--sizes', nargs='+', type=int, default=[100_000, 500_000, 1_000_000, 2_000_000])
    args = ap.parse_args()
    if sorted(args.sizes) != args.sizes or min(args.sizes) <= 0:
        ap.error('sizes must be increasing positive integers')
    raw = Path('data/raw'); out = Path('data/prepared')
    download(ZONES, raw/'taxi_zone_lookup.csv')
    months = []
    for month in args.months:
        f = raw/f'yellow_tripdata_{month}.parquet'
        download(URL.format(month=month), f)
        months.append(f)
    # Preserve original row order and use prefixes of the same underlying data.
    import pyarrow.dataset as ds
    dataset = ds.dataset([str(x) for x in months], format='parquet')
    missing = set(COLS) - set(dataset.schema.names)
    if missing: raise ValueError(f'Missing columns: {missing}')
    reader = dataset.scanner(columns=COLS, batch_size=65536).to_batches()
    chunks = []; total = 0; sizes = iter(args.sizes); target = next(sizes, None)
    for batch in reader:
        if target is None: break
        chunks.append(batch); total += batch.num_rows
        while target is not None and total >= target:
            table = pa.Table.from_batches(chunks).slice(0, target)
            dest = out/f'trips_{target}.parquet'; dest.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(table, dest, row_group_size=100_000)
            print(f'Created {dest} ({target} rows)')
            target = next(sizes, None)
    if target is not None: raise ValueError(f'Only {total} rows available; cannot make {target}')

if __name__ == '__main__': main()
