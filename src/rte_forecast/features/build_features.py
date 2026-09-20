"""Construction des features SANS fuite de données.

Convention temporelle (documentée dans le README) :
    par défaut (``issue_hour=None``) la prévision du jour D est émise à D 00:00 ; la dernière
    charge connue est celle de D-1 23:00.
    => toute feature dérivée de la charge pour l'heure t doit n'utiliser que des observations
       d'instant <= floor_day(t) - 1h, soit un retard minimal de 24 h (``min_lag_hours``).

    Avec ``issue_hour=k`` (0..22) la prévision de D est émise à D-1 (k+1):00 : la dernière charge
    connue est celle de D-1 k:00 (ex. k=9 -> émission à D-1 10:00, comme une vraie J-1). Pour
    l'heure cible h, un retard L n'est légal que si L >= 24 + h - k : le « lag 24 h » devient un
    lag de 48 h pour les heures h > k, et les moyennes glissantes sont ancrées à D-1 k:00.

Deux variantes des features autorégressives :

* ``clean_lags=False`` : retards bruts (modèles « tous jours »).
* ``clean_lags=True``  : les retards dont la SOURCE est un jour atypique (férié, pont, période
  spéciale) sont neutralisés, pour que le modèle « jour normal » n'apprenne pas de fêtes. Pour
  les retards hebdomadaires on remonte à la semaine normale la plus récente (<= max_fallback_weeks).
  Sinon NaN (gérés nativement par XGBoost et scikit-learn RandomForest).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEK = 168

CALENDAR_COLS = ["hour", "weekday", "month", "week_of_year", "is_weekend", "is_public_holiday",
                 "is_bridge_day", "is_ramp_up", "is_ramp_down", "is_august_special_period",
                 "prev_day_is_holiday_or_bridge", "next_day_is_holiday_or_bridge"]
WEATHER_NORMAL_MODE = ["temperature_forecast", "temperature_forecast_squared", "temp_x_weekend",
                       "temp_x_hour", "temp_sq_x_weekend"]
WEATHER_PERFECT_EXTRA = ["temperature_realised", "temperature_realised_squared",
                         "temperature_normal", "temperature_normal_squared"]
LOAD_COLS = ["load_lag_24h", "load_lag_48h", "load_lag_168h", "load_lag_336h",
             "rolling_mean_24h", "rolling_mean_7d"]


WEATHER_MODES = ("normal", "noisy", "perfect")


def feature_columns(weather_mode: str = "normal") -> list[str]:
    """Colonnes de features d'un modèle selon le mode météo.

    * ``normal``  : prévision de température = normale Enedis (mode opérationnel de référence) ;
    * ``noisy``   : prévision = réalisée + erreur de prévision SYNTHÉTIQUE (scénario réaliste) ;
    * ``perfect`` : prévision = réalisée (oracle, plafond de gain) + la réalisée en feature.
    """
    if weather_mode not in WEATHER_MODES:
        raise ValueError(f"weather_mode doit appartenir à {WEATHER_MODES}")
    cols = CALENDAR_COLS + WEATHER_NORMAL_MODE + LOAD_COLS
    if weather_mode == "perfect":
        cols = cols + WEATHER_PERFECT_EXTRA
    return cols


def make_temperature_forecast(df: pd.DataFrame, weather_mode: str = "normal", sigma: float = 1.2,
                              phi: float = 0.6, seed: int = 42) -> pd.Series:
    """Température « prévue » utilisée comme feature selon le mode météo (voir feature_columns).

    Le mode ``noisy`` est SYNTHÉTIQUE : réalisée + anomalie AR(1) journalière (σ=1.2 °C,
    φ=0.6) ; à remplacer par de vraies prévisions archivées dès qu'elles seront disponibles.
    """
    if weather_mode == "normal":
        return df["temperature_normal"]
    if weather_mode == "perfect":
        return df["temperature_realised"]
    if weather_mode == "noisy":
        from rte_forecast.data.temperature import synthetic_anomaly
        return df["temperature_realised"] + synthetic_anomaly(df.index, sigma, phi, seed)
    raise ValueError(f"weather_mode doit appartenir à {WEATHER_MODES}")


def check_issue_hour(issue_hour: int | None) -> int:
    """Dernière heure de D-1 dont la charge est connue à l'émission (23 = D 00:00, historique)."""
    if issue_hour is None:
        return 23
    if not 0 <= int(issue_hour) <= 22:
        raise ValueError("issue_hour doit valoir None ou un entier entre 0 et 22")
    return int(issue_hour)


def _issue_lag(series: pd.Series, hours: np.ndarray, known: int, base_lag: int) -> pd.Series:
    """Retard `base_lag` h si la source est connue à l'émission, sinon `base_lag` + 24 h."""
    return pd.Series(np.where(hours <= known, series.shift(base_lag), series.shift(base_lag + 24)),
                     index=series.index)


def _anchored_rolling(series: pd.Series, index: pd.DatetimeIndex, window: int, known: int,
                      min_periods: int) -> pd.Series:
    """Moyenne glissante de `window` h se terminant à la dernière charge connue (D-1 known:00)."""
    anchor = index.floor("D") - pd.Timedelta(hours=24 - known)
    return pd.Series(series.rolling(window, min_periods=min_periods).mean().reindex(anchor)
                     .to_numpy(), index=index)


def _lag_with_fallback(series: pd.Series, atypical: pd.Series, start_weeks: int,
                       max_weeks: int) -> pd.Series:
    """Retard hebdomadaire ; si la source est atypique, on remonte de semaine en semaine."""
    out = series.shift(WEEK * start_weeks).copy()
    bad = atypical.shift(WEEK * start_weeks).fillna(True).astype(bool)
    for k in range(start_weeks + 1, max_weeks + 1):
        cand = series.shift(WEEK * k)
        cand_bad = atypical.shift(WEEK * k).fillna(True).astype(bool)
        take = bad & ~cand_bad
        out = out.where(~take, cand)
        bad = bad & cand_bad
    return out.where(~bad, np.nan)


def build_features(df: pd.DataFrame, cal: pd.DataFrame, *, weather_mode: str = "normal",
                   min_lag_hours: int = 24, clean_lags: bool = False,
                   max_fallback_weeks: int = 6,
                   temperature_forecast: pd.Series | None = None,
                   issue_hour: int | None = None) -> pd.DataFrame:
    """Retourne le DataFrame de features + `load_mw` (cible) + `special_period_type`.

    `df` : index horaire régulier, colonnes `load_mw`, `temperature_normal`,
    `temperature_realised`. `cal` : calendrier horaire de `build_hourly_calendar`.
    `temperature_forecast` : scénario de température prévue imposé (sinon dérivé de `weather_mode`).
    `issue_hour` : voir le docstring du module (None = émission à D 00:00, comportement historique).
    """
    if min_lag_hours < 24:
        raise ValueError("min_lag_hours < 24 introduirait de la fuite (prévision émise à D 00:00)")
    known = check_issue_hour(issue_hour)
    hours = df.index.hour.to_numpy()
    load = df["load_mw"]
    X = pd.DataFrame(index=df.index)
    X["hour"] = cal["hour"]
    X["weekday"] = cal["weekday"]
    X["month"] = cal["month"]
    X["week_of_year"] = cal["iso_week"]
    for c in ["is_weekend", "is_public_holiday", "is_bridge_day", "is_ramp_up", "is_ramp_down",
              "is_august_special_period"]:
        X[c] = cal[c].astype(int)

    off = (cal["is_public_holiday"] | cal["is_bridge_day"])
    X["prev_day_is_holiday_or_bridge"] = off.shift(24, fill_value=False).astype(int)
    X["next_day_is_holiday_or_bridge"] = off.shift(-24, fill_value=False).astype(int)

    # Météo : la « prévision » de température est la normale (mode opérationnel) ; en mode
    # « perfect » on triche volontairement avec la réalisée pour mesurer le plafond de gain.
    t_norm, t_real = df["temperature_normal"], df["temperature_realised"]
    t_fc = temperature_forecast if temperature_forecast is not None \
        else make_temperature_forecast(df, weather_mode)
    X["temperature_forecast"] = t_fc
    X["temperature_forecast_squared"] = t_fc**2
    X["temp_x_weekend"] = t_fc * X["is_weekend"]
    X["temp_x_hour"] = t_fc * X["hour"]
    X["temp_sq_x_weekend"] = X["temperature_forecast_squared"] * X["is_weekend"]
    X["temperature_realised"] = t_real
    X["temperature_realised_squared"] = t_real**2
    X["temperature_normal"] = t_norm
    X["temperature_normal_squared"] = t_norm**2

    if clean_lags:
        atypical = ~cal["is_normal_day"].astype(bool)
        masked = load.where(~atypical)
        X["load_lag_168h"] = _lag_with_fallback(load, atypical, 1, max_fallback_weeks)
        X["load_lag_336h"] = _lag_with_fallback(load, atypical, 2, max_fallback_weeks)
        if issue_hour is None:
            X["load_lag_24h"] = masked.shift(min_lag_hours)
            X["load_lag_48h"] = masked.shift(min_lag_hours + 24)
            # fenêtres glissantes sur la série masquée, décalées de min_lag_hours
            past = masked.shift(min_lag_hours)
            X["rolling_mean_24h"] = past.rolling(24, min_periods=12).mean()
            X["rolling_mean_7d"] = past.rolling(24 * 7, min_periods=48).mean()
        else:
            X["load_lag_24h"] = _issue_lag(masked, hours, known, 24)
            X["load_lag_48h"] = _issue_lag(masked, hours, known, 48)
            X["rolling_mean_24h"] = _anchored_rolling(masked, df.index, 24, known, 12)
            X["rolling_mean_7d"] = _anchored_rolling(masked, df.index, 24 * 7, known, 48)
    else:
        X["load_lag_168h"] = load.shift(WEEK)
        X["load_lag_336h"] = load.shift(2 * WEEK)
        if issue_hour is None:
            X["load_lag_24h"] = load.shift(min_lag_hours)
            X["load_lag_48h"] = load.shift(min_lag_hours + 24)
            past = load.shift(min_lag_hours)
            X["rolling_mean_24h"] = past.rolling(24).mean()
            X["rolling_mean_7d"] = past.rolling(24 * 7).mean()
        else:
            X["load_lag_24h"] = _issue_lag(load, hours, known, 24)
            X["load_lag_48h"] = _issue_lag(load, hours, known, 48)
            X["rolling_mean_24h"] = _anchored_rolling(load, df.index, 24, known, 24)
            X["rolling_mean_7d"] = _anchored_rolling(load, df.index, 24 * 7, known, 24 * 7)

    X["special_period_type"] = cal["special_period_type"]
    X["load_mw"] = load
    return X
