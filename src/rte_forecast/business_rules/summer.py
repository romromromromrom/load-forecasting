"""Moteur « été » : creux autour du 15 août (ramp-down -> point bas -> ramp-up).

Alignement par la « semaine du creux » (week-end le plus proche du 15 août), pas par le
calendrier civil. Chaîne hiérarchique :

    niveau normal avant vacances (3 semaines précédant la fenêtre)      = summer_start_level
        x w(k)   forme hebdomadaire moyenne (médiane des années de référence), k = semaines
                 relatives au creux                                      -> niveau hebdomadaire
        x c(j)   coefficient jour de semaine (médiane, mois normaux)     -> niveau journalier
        x p(j,h) profil horaire normalisé (médiane)                      -> prévision horaire

Les niveaux hebdomadaires sont « désaisonnalisés » du jour de semaine et privés des fériés/ponts
(sinon l'effet du 15 août serait compté deux fois : dans w(k) puis dans le coefficient férié).
Le 15 août lui-même (et un éventuel pont) passe par ``isolated_holidays``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from rte_forecast.business_rules import common
from rte_forecast.business_rules import isolated_holidays as ih
from rte_forecast.business_rules.weather import WeatherNormalizer
from rte_forecast.calendar import special_periods as sp

SUMMER_METHOD = "SUMMER_BUSINESS_RULE"


def _valid_day(cal_daily: pd.DataFrame, d: dt.date) -> bool:
    r = cal_daily.loc[pd.Timestamp(d)]
    return not (r["is_public_holiday"] or r["is_bridge_day"])


def window_geometry(year: int, before: int, after: int) -> dict:
    trough = sp.summer_trough_week(year)
    start, end = sp.summer_window(year, before, after)
    return {"trough_monday": trough, "start": start, "end": end}


def _week_level(load: pd.Series, monday: dt.date, cal_daily: pd.DataFrame, coef: dict[int, float]
                ) -> float | None:
    """Niveau hebdomadaire désaisonnalisé : moyenne des (jour / c[jour de semaine]) valides."""
    vals = []
    for i in range(7):
        d = monday + dt.timedelta(days=i)
        v = common.day_values(load, d)
        if v is None or not _valid_day(cal_daily, d):
            continue
        vals.append(v.mean() / coef[d.weekday()])
    return float(np.mean(vals)) if len(vals) >= 3 else None


def _weekday_coefficients(load: pd.Series, mondays: list[dt.date], cal_daily: pd.DataFrame
                          ) -> dict[int, float]:
    """c[j] = médiane de (moyenne du jour / moyenne de sa semaine) sur les semaines 100 % valides."""
    acc: dict[int, list[float]] = {j: [] for j in range(7)}
    for m in mondays:
        days = [m + dt.timedelta(days=i) for i in range(7)]
        if all(common.day_values(load, d) is not None and _valid_day(cal_daily, d) for d in days):
            wm = common.week_mean(load, m)
            for d in days:
                acc[d.weekday()].append(common.day_mean(load, d) / wm)
    if not all(acc.values()):
        return {j: 1.0 for j in range(7)}
    med = {j: float(np.median(v)) for j, v in acc.items()}
    scale = 7.0 / sum(med.values())
    return {j: v * scale for j, v in med.items()}


def forecast_summer(load: pd.Series, target_year: int, cal_daily: pd.DataFrame,
                    stats: pd.DataFrame, before: int = 2, after: int = 2, anchor_weeks: int = 3,
                    covid_years=(2020,), normalizer: WeatherNormalizer | None = None,
                    level_mode: str = "planning", recent_lag_days: int = 0) -> pd.DataFrame:
    """Prévision horaire de la fenêtre estivale de `target_year`.

    ``planning`` : émise au premier jour de la fenêtre, niveau = moyenne des 3 semaines d'avant ;
    ``adaptive`` : niveau ré-ancré chaque jour sur les 7 jours précédents réellement observés,
    rapporté à la forme moyenne (mêmes coefficients de forme, seul le niveau change) ;
    ``recent_lag_days`` = 1 décale ces 7 jours d'un jour (émission à J-1 10:00, J-1 incomplet).
    """
    if level_mode not in {"planning", "adaptive"}:
        raise ValueError("level_mode doit valoir 'planning' ou 'adaptive'")
    geo = window_geometry(target_year, before, after)
    origin = pd.Timestamp(geo["start"])
    history = load.loc[: origin - pd.Timedelta(hours=1)]
    g = None
    if normalizer is not None:
        history, g = normalizer.adjust(history)
    full_adj = load if normalizer is None else normalizer.adjust_with(load, g)

    def anchor_mondays(start: dt.date) -> list[dt.date]:
        return [start - dt.timedelta(weeks=j) for j in range(anchor_weeks, 0, -1)]

    # années de référence : fenêtre + ancre complètes dans l'historique connu
    refs = []
    for r in range(load.index.min().year, target_year):
        gr = window_geometry(r, before, after)
        days = [gr["start"] - dt.timedelta(weeks=anchor_weeks) + dt.timedelta(days=i)
                for i in range((gr["end"] - gr["start"]).days + 1 + 7 * anchor_weeks)]
        if common.is_complete(history, days):
            refs.append(r)
    if not refs:
        raise ValueError(f"aucune année de référence estivale pour {target_year}")
    pool = [r for r in refs if r not in set(covid_years)] or refs

    all_mondays: list[dt.date] = []
    for r in pool:
        gr = window_geometry(r, before, after)
        all_mondays += anchor_mondays(gr["start"]) + [
            gr["trough_monday"] + dt.timedelta(weeks=k) for k in range(-before, after + 1)]
    coef = _weekday_coefficients(history, all_mondays, cal_daily)

    ratios = {k: [] for k in range(-before, after + 1)}
    profiles: dict[int, list[np.ndarray]] = {j: [] for j in range(7)}
    for r in pool:
        gr = window_geometry(r, before, after)
        anchors = [_week_level(history, m, cal_daily, coef) for m in anchor_mondays(gr["start"])]
        anchors = [a for a in anchors if a is not None]
        if not anchors:
            continue
        a_r = float(np.mean(anchors))
        for k in ratios:
            m = gr["trough_monday"] + dt.timedelta(weeks=k)
            lv = _week_level(history, m, cal_daily, coef)
            if lv is not None:
                ratios[k].append(lv / a_r)
            for i in range(7):
                d = m + dt.timedelta(days=i)
                if _valid_day(cal_daily, d) and common.day_values(history, d) is not None:
                    profiles[d.weekday()].append(common.normalized_hourly_profile(history, d))
    shape = {k: float(np.median(v)) for k, v in ratios.items() if v}
    prof = {j: (np.median(v, axis=0) / np.median(v, axis=0).mean()) for j, v in profiles.items() if v}

    anchors_t = [_week_level(history, m, cal_daily, coef) for m in anchor_mondays(geo["start"])]
    anchors_t = [a for a in anchors_t if a is not None]
    if not anchors_t or len(shape) < before + after + 1 or len(prof) < 7:
        raise ValueError(f"référence estivale incomplète pour {target_year}")
    start_level = float(np.mean(anchors_t))
    levels = {"summer_start_level_mw": start_level,
              "summer_trough_level_mw": start_level * shape[0],
              "summer_end_level_mw": start_level * shape[after]}

    def shape_at(d: dt.date) -> float:
        k_ = (common.monday_of(d) - geo["trough_monday"]).days // 7
        return 1.0 if k_ < -before else shape[min(k_, after)]

    def recent_level(day: dt.date) -> tuple[float, float] | None:
        """(niveau désaisonnalisé réel, forme moyenne correspondante) des 7 jours précédents."""
        hist_d = full_adj.loc[: pd.Timestamp(day) - pd.Timedelta(days=recent_lag_days,
                                                                  hours=1)]
        lv, sh = [], []
        for j in range(1 + recent_lag_days, 8 + recent_lag_days):
            x = day - dt.timedelta(days=j)
            if _valid_day(cal_daily, x) and common.day_values(hist_d, x) is not None:
                lv.append(common.day_mean(hist_d, x) / coef[x.weekday()])
                sh.append(shape_at(x))
        return (float(np.mean(lv)), float(np.mean(sh))) if len(lv) >= 3 else None

    rows = []
    n_days = (geo["end"] - geo["start"]).days + 1
    for i in range(n_days):
        d = geo["start"] + dt.timedelta(days=i)
        k = (common.monday_of(d) - geo["trough_monday"]).days // 7
        wd = d.weekday()
        weekly_level = start_level * shape[k]
        if level_mode == "adaptive":
            rec = recent_level(d)
            if rec is not None:
                weekly_level = rec[0] * shape[k] / rec[1]
        day_level = weekly_level * coef[wd]
        hourly = day_level * prof[wd]
        method, reason, day_coef = SUMMER_METHOD, (
            f"années de référence {pool} ; semaine relative au creux k={k:+d}"), coef[wd]
        source = "médiane des profils estivaux (jour de semaine identique)"
        r = cal_daily.loc[pd.Timestamp(d)]
        if r["is_public_holiday"] or r["is_bridge_day"]:
            res = ih.forecast_isolated_holiday(pd.Timestamp(d), hourly, stats, history, r)
            if res is not None:
                hourly, day_level = res["yhat"], res["day_mean_mw"]
                method = SUMMER_METHOD + "+" + res["method"]
                reason += " ; " + res["reason"]
                day_coef, source = res["day_coefficient"], res["hourly_profile_source"]
        for h in range(24):
            rows.append({
                "timestamp": pd.Timestamp(d) + pd.Timedelta(hours=h), "yhat": float(hourly[h]),
                "method": method, "reference_year": ",".join(map(str, pool)), "reason": reason,
                "weekly_target_level_mw": weekly_level, "day_coefficient": day_coef,
                "day_mean_mw": day_level, "hourly_profile_source": source, "hour": h, **levels})
    out = pd.DataFrame(rows).set_index("timestamp")
    if g is not None:
        out["yhat"] = out["yhat"] + normalizer.forecast_correction(out.index, g)
    out.attrs.update({"shape": shape, "coefficients": coef, "origin": origin, "geometry": geo})
    return out
