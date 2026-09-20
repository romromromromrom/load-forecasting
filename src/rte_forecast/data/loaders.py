"""Lecture des sources brutes : eCO2mix (RTE), température Enedis, Open-Meteo (backfill)."""

from __future__ import annotations

import io
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
import requests

ECO2MIX_HEAD = ["perimetre", "nature", "date", "heure", "consommation_mw",
                "prevision_j1_mw", "prevision_j_mw"]

# (nom, latitude, longitude, poids ~ population régionale en millions) pour la température
# nationale pondérée servant à boucher le trou Enedis. Proxy, pas la méthode Enedis.
CITIES = [
    ("Paris", 48.85, 2.35, 12.3), ("Lyon", 45.76, 4.84, 8.1), ("Marseille", 43.30, 5.37, 5.1),
    ("Toulouse", 43.60, 1.44, 5.9), ("Lille", 50.63, 3.06, 6.0), ("Bordeaux", 44.84, -0.58, 6.0),
    ("Nantes", 47.22, -1.55, 3.9), ("Strasbourg", 48.57, 7.75, 5.6), ("Rennes", 48.11, -1.68, 3.4),
    ("Dijon", 47.32, 5.04, 2.8), ("Orleans", 47.90, 1.91, 2.6), ("Rouen", 49.44, 1.10, 3.3),
]


def _slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()


def read_eco2mix_zip(path: str | Path) -> pd.DataFrame:
    """Lit un zip annuel eCO2mix (TSV latin-1 déguisé en .xls) au pas natif 15 min.

    Le fichier contient une ligne de disclaimer finale et une colonne vide finale ; la
    consommation n'est renseignée qu'aux pas :00 et :30 (les :15/:45 sont vides).
    """
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), sep="\t", encoding="latin-1", index_col=False, dtype=str)
    df.columns = ECO2MIX_HEAD + [_slug(c) for c in df.columns[len(ECO2MIX_HEAD):]]
    df = df[df["date"].fillna("").str.match(r"^\d{4}-\d{2}-\d{2}$")].copy()  # exclut le disclaimer
    for col in df.columns.difference(["perimetre", "nature", "date", "heure"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.insert(0, "timestamp", pd.to_datetime(df["date"] + " " + df["heure"]))
    return df.drop(columns=["date", "heure"]).reset_index(drop=True)


def load_eco2mix(zip_dir: str | Path, pattern: str) -> pd.DataFrame:
    files = sorted(Path(zip_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier eCO2mix '{pattern}' dans {zip_dir}")
    frames = [read_eco2mix_zip(f).assign(source_file=f.name) for f in files]
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def fetch_enedis_temperature(api_url: str, timeout: int = 180) -> pd.DataFrame:
    """Télécharge tout le dataset Enedis (export CSV). Horodates en UTC."""
    params = {
        "select": "horodate,temperature_realisee_lissee_degc,temperature_normale_lissee_degc,"
                  "pseudo_rayonnement",
        "order_by": "horodate", "delimiter": ";",
    }
    r = requests.get(f"{api_url}/exports/csv", params=params, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), sep=";")
    df["horodate"] = pd.to_datetime(df["horodate"], utc=True)
    return df.rename(columns={
        "horodate": "timestamp_utc", "temperature_realisee_lissee_degc": "temperature_realised",
        "temperature_normale_lissee_degc": "temperature_normal",
        "pseudo_rayonnement": "pseudo_radiation"})


def _get_json_with_retry(url: str, params: dict, timeout: int, tries: int = 4) -> dict:
    """GET JSON avec backoff exponentiel (l'API gratuite Open-Meteo limite le débit)."""
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2**attempt)
    raise RuntimeError("unreachable")


def fetch_openmeteo_cities(api_url: str, start: str, end: str, timeout: int = 120) -> pd.DataFrame:
    """Température 2 m horaire ERA5 (Open-Meteo archive) pour quelques villes, en UTC."""
    frames = []
    for name, lat, lon, weight in CITIES:
        h = _get_json_with_retry(api_url, {
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "hourly": "temperature_2m", "timezone": "UTC"}, timeout)["hourly"]
        frames.append(pd.DataFrame({
            "timestamp_utc": pd.to_datetime(h["time"], utc=True),
            "temperature_2m": h["temperature_2m"], "city": name, "weight": weight}))
    return pd.concat(frames, ignore_index=True)
