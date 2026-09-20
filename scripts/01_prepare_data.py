"""Étape 1 : sources brutes -> Parquet brut -> Parquet préparé horaire."""

import sys

from rte_forecast.data.pipeline import build_hourly_dataset

if __name__ == "__main__":
    df = build_hourly_dataset(refresh="--refresh" in sys.argv)
    print(df.shape, df.index.min(), "->", df.index.max())
    print(df["temperature_source"].value_counts())
