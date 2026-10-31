"""Two joins: zone lookup, then a derived grouped table keyed by dropoff zone."""

import pandas as pd
import dask.dataframe as dd
import dask


def build(path, zones_path, partitions):
    trips = dd.read_parquet(
        path,
        columns=[
            "PULocationID",
            "DOLocationID",
            "fare_amount",
            "trip_distance",
        ],
    ).repartition(npartitions=partitions)

    zones = pd.read_csv(zones_path)[
        ["LocationID", "Borough"]
    ].rename(columns={"LocationID": "PULocationID"})

    zones["PULocationID"] = zones["PULocationID"].astype(
        trips._meta["PULocationID"].dtype
    )

    # First join: small lookup table.
    enriched = trips.merge(
        zones,
        on="PULocationID",
        how="left",
    )

    # Second join: derived table grouped by dropoff location.
    derived = (
        trips.groupby("DOLocationID")
        .agg(
            {
                "fare_amount": "mean",
                "trip_distance": "mean",
            },
            split_out=partitions,
        )
        .reset_index()
    )

    derived = derived.rename(
        columns={
            "fare_amount": "zone_mean_fare",
            "trip_distance": "zone_mean_distance",
        }
    )

    return enriched.merge(
        derived,
        on="DOLocationID",
        how="left",
    )


def run(path, zones_path, partitions):
    df = build(path, zones_path, partitions)

    fare_sum, zone_sum, row_count = dask.compute(
        df["fare_amount"].sum(),
        df["zone_mean_fare"].sum(),
        df["fare_amount"].count(),
    )

    return {
        "rows": int(row_count),
        "checksum": float(fare_sum),
    }
