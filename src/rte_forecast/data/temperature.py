"""Construction de la série horaire de température (réalisée + normale) en heure locale.

Enedis fournit la température nationale lissée (réalisée et normale) mais seulement sur une
fenêtre glissante de 5 ans. Le trou en amont est bouché, par ordre de préférence, par :

1. ``openmeteo_calibrated`` : ERA5 (Open-Meteo) pondéré sur 12 villes, lissé puis recalé
   linéairement sur Enedis via la période de recouvrement (donnée réelle, mais proxy) ;
2. ``synthetic`` : normale climatologique + anomalie AR(1) aléatoire (SYNTHÉTIQUE, à remplacer).

Chaque ligne porte ``temperature_source`` pour que rien ne soit confondu avec de la donnée Enedis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "Europe/Paris"
SRC_ENEDIS = "enedis"
SRC_OPENMETEO = "openmeteo_calibrated"
SRC_SYNTHETIC = "synthetic"


def enedis_to_local_hourly(enedis: pd.DataFrame) -> pd.DataFrame:
    """UTC 15/30 min -> heure locale naïve (mur d'horloge Europe/Paris), pas horaire.

    Les heures dupliquées du changement d'heure d'automne sont moyennées ; l'heure manquante
    du printemps est interpolée linéairement.
    """
    df = enedis.copy()
    local = df["timestamp_utc"].dt.tz_convert(TZ).dt.tz_localize(None)
    df["timestamp"] = local.dt.floor("h")
    cols = ["temperature_realised", "temperature_normal", "pseudo_radiation"]
    hourly = df.groupby("timestamp")[cols].mean()
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    return hourly.reindex(full).interpolate(limit=3).rename_axis("timestamp")


def openmeteo_national(om: pd.DataFrame) -> pd.Series:
    """Moyenne pondérée des villes, indexée en heure locale naïve."""
    w = om.assign(wt=om["temperature_2m"] * om["weight"])
    g = w.groupby("timestamp_utc").agg(wt=("wt", "sum"), w=("weight", "sum"))
    s = (g["wt"] / g["w"]).rename("t_om")
    idx = s.index.tz_convert(TZ).tz_localize(None)
    s.index = idx
    return s[~s.index.duplicated(keep="first")].sort_index().asfreq("h").interpolate(limit=3)


def calibrate_openmeteo(om_hourly: pd.Series, enedis_real: pd.Series,
                        halflives=(0, 3, 6, 12, 24, 36, 48)) -> dict:
    """Ajuste enedis ≈ a + b * EWMA(om, demi-vie h) sur le recouvrement ; retient le meilleur h."""
    best = None
    for hl in halflives:
        sm = om_hourly if hl == 0 else om_hourly.ewm(halflife=hl, adjust=False).mean()
        j = pd.concat([sm.rename("x"), enedis_real.rename("y")], axis=1).dropna()
        b, a = np.polyfit(j["x"], j["y"], 1)
        rmse = float(np.sqrt(np.mean((j["y"] - (a + b * j["x"])) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = {"halflife": hl, "a": float(a), "b": float(b), "rmse": rmse, "n": len(j),
                    "corr": float(np.corrcoef(j["x"], j["y"])[0, 1])}
    return best


def apply_calibration(om_hourly: pd.Series, cal: dict) -> pd.Series:
    sm = om_hourly if cal["halflife"] == 0 else om_hourly.ewm(halflife=cal["halflife"],
                                                              adjust=False).mean()
    return cal["a"] + cal["b"] * sm


def climatological_normal(enedis_hourly: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Normale Enedis moyenne par (mois, jour, heure), projetée sur `index` (29 fév. -> 28 fév.)."""
    src = enedis_hourly["temperature_normal"].dropna()
    key = pd.MultiIndex.from_arrays([src.index.month, src.index.day, src.index.hour])
    clim = src.groupby(key).mean()
    d = index.day.where(~((index.month == 2) & (index.day == 29)), 28)
    q = pd.MultiIndex.from_arrays([index.month, d, index.hour])
    return pd.Series(clim.reindex(q).to_numpy(), index=index)


def synthetic_anomaly(index: pd.DatetimeIndex, sigma: float, phi_daily: float, seed: int) -> pd.Series:
    """Anomalie SYNTHÉTIQUE AR(1) quotidienne (aucun lien avec la météo réelle)."""
    rng = np.random.default_rng(seed)
    days = index.normalize().unique()
    eps = rng.normal(0, sigma * np.sqrt(1 - phi_daily**2), len(days))
    a = np.zeros(len(days))
    for i in range(1, len(days)):
        a[i] = phi_daily * a[i - 1] + eps[i]
    return pd.Series(a, index=days).reindex(index.normalize()).set_axis(index)


def build_temperature_hourly(enedis_raw: pd.DataFrame, target_index: pd.DatetimeIndex,
                             backfill: str = "openmeteo_calibrated",
                             om_raw: pd.DataFrame | None = None, seed: int = 42):
    """Retourne (DataFrame horaire aligné sur target_index, dict de diagnostics de calibration)."""
    enedis = enedis_to_local_hourly(enedis_raw)
    out = pd.DataFrame(index=target_index)
    out["temperature_realised"] = enedis["temperature_realised"].reindex(target_index)
    out["temperature_normal"] = enedis["temperature_normal"].reindex(target_index)
    out["pseudo_radiation"] = enedis["pseudo_radiation"].reindex(target_index)
    out["temperature_source"] = np.where(out["temperature_realised"].notna(), SRC_ENEDIS, "")
    diag: dict = {"enedis_first": str(enedis.index.min()), "enedis_last": str(enedis.index.max())}

    gap = out["temperature_realised"].isna()
    diag["gap_hours"] = int(gap.sum())
    if not gap.any():
        return out, diag

    # La normale climatologique ne dépend pas de l'année : on la projette sur le trou.
    out.loc[gap, "temperature_normal"] = climatological_normal(enedis, target_index)[gap]

    if backfill == "openmeteo_calibrated" and om_raw is not None and len(om_raw):
        om_h = openmeteo_national(om_raw)
        cal = calibrate_openmeteo(om_h, enedis["temperature_realised"])
        proxy = apply_calibration(om_h, cal).reindex(target_index)
        fill = gap & proxy.notna()
        out.loc[fill, "temperature_realised"] = proxy[fill]
        out.loc[fill, "temperature_source"] = SRC_OPENMETEO
        diag["calibration"] = cal
        gap = out["temperature_realised"].isna()

    if gap.any():  # dernier recours : synthétique, clairement marqué
        anomaly = synthetic_anomaly(target_index, sigma=3.0, phi_daily=0.8, seed=seed)
        out.loc[gap, "temperature_realised"] = (out["temperature_normal"] + anomaly)[gap]
        out.loc[gap, "temperature_source"] = SRC_SYNTHETIC
        diag["synthetic_hours"] = int(gap.sum())

    out["pseudo_radiation"] = out["pseudo_radiation"].interpolate(limit=3)
    out["temperature_normal"] = out["temperature_normal"].interpolate(limit=3)
    return out, diag
