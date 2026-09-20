"""Orchestration : sources brutes -> Parquet brut -> Parquet préparé (horaire, heure locale)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from rte_forecast.config import load_config, resolve_path
from rte_forecast.data import cleaning, imputation, loaders, temperature


def fetch_raw(cfg: dict, refresh: bool = False) -> dict[str, Path]:
    """Télécharge/lit les sources et les écrit en Parquet dans data/raw (cache si présent)."""
    raw = resolve_path(cfg, "raw_dir")
    raw.mkdir(parents=True, exist_ok=True)
    paths = {"eco2mix": raw / "eco2mix_raw_15min.parquet",
             "enedis": raw / "enedis_temperature_raw.parquet",
             "openmeteo": raw / "openmeteo_cities_raw.parquet"}
    if refresh or not paths["eco2mix"].exists():
        eco = loaders.load_eco2mix(resolve_path(cfg, "eco2mix_zip_dir"),
                                   cfg["paths"]["eco2mix_zip_glob"])
        eco.to_parquet(paths["eco2mix"], index=False)
    if refresh or not paths["enedis"].exists():
        loaders.fetch_enedis_temperature(cfg["temperature"]["enedis_api"]).to_parquet(
            paths["enedis"], index=False)
    if cfg["temperature"]["backfill"] == "openmeteo_calibrated" and (
            refresh or not paths["openmeteo"].exists()):
        eco_ts = pd.read_parquet(paths["eco2mix"], columns=["timestamp"])["timestamp"]
        start = (eco_ts.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        end = (eco_ts.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            loaders.fetch_openmeteo_cities(cfg["temperature"]["openmeteo_api"], start, end
                                           ).to_parquet(paths["openmeteo"], index=False)
        except Exception as exc:  # réseau indisponible -> repli synthétique, clairement marqué
            warnings.warn(f"Open-Meteo indisponible ({exc}) : backfill SYNTHÉTIQUE utilisé.",
                          stacklevel=2)
    return paths


def build_hourly_dataset(cfg: dict | None = None, refresh: bool = False) -> pd.DataFrame:
    """Construit et sauvegarde `data/processed/hourly.parquet` (+ rapport de qualité JSON)."""
    cfg = cfg or load_config()
    paths = fetch_raw(cfg, refresh=refresh)
    eco = pd.read_parquet(paths["eco2mix"])
    hourly = cleaning.aggregate_hourly(eco)

    invalid = cleaning.detect_invalid(hourly["load_mw"])
    hourly["load_raw_mw"] = hourly["load_mw"]
    hourly["load_imputed"] = invalid
    hourly["load_mw"] = imputation.impute_causal(hourly["load_mw"], invalid)
    cleaning.validate_hourly(hourly)

    om = pd.read_parquet(paths["openmeteo"]) if paths["openmeteo"].exists() else None
    temp, diag = temperature.build_temperature_hourly(
        pd.read_parquet(paths["enedis"]), hourly.index,
        backfill=cfg["temperature"]["backfill"], om_raw=om,
        seed=cfg["temperature"]["synthetic_seed"])
    hourly = hourly.join(temp)
    hourly["is_synthetic_temperature"] = hourly["temperature_source"].eq(temperature.SRC_SYNTHETIC)

    out_dir = resolve_path(cfg, "processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out_dir / "hourly.parquet")
    quality = {
        "rows": len(hourly), "start": str(hourly.index.min()), "end": str(hourly.index.max()),
        "invalid_load_hours": int(invalid.sum()),
        "temperature_source_hours": hourly["temperature_source"].value_counts().to_dict(),
        "temperature": diag,
        "load_range_mw": [float(hourly["load_mw"].min()), float(hourly["load_mw"].max())],
    }
    (out_dir / "data_quality.json").write_text(json.dumps(quality, indent=2, default=str))
    return hourly


def load_hourly(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(resolve_path(cfg, "processed_dir") / "hourly.parquet")
