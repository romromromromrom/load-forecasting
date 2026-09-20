"""Vérification empirique des hypothèses métier (rien n'est corrigé silencieusement).

Chaque fonction retourne un DataFrame lisible ; les exceptions à une hypothèse sont
signalées dans une colonne `verdict`, jamais masquées.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from rte_forecast.business_rules import common
from rte_forecast.business_rules import isolated_holidays as ih
from rte_forecast.calendar import special_periods as sp
from rte_forecast.data import imputation


def _weekly(load: pd.Series) -> pd.Series:
    w = load.groupby(load.index.to_period("W-SUN")).mean()
    w.index = w.index.start_time
    return w


def summer_trough_check(load: pd.Series, years: list[int]) -> pd.DataFrame:
    """H1 : le creux estival tombe la semaine dont le week-end est le plus proche du 15 août."""
    wk = _weekly(load)
    rows = []
    for y in years:
        cand = pd.Timestamp(sp.summer_trough_week(y))
        for label, lo, hi in (("juin–sept", f"{y}-06-01", f"{y}-09-30"),
                              ("juil–sept", f"{y}-07-01", f"{y}-09-30")):
            s = wk.loc[lo:hi]
            s = s[[t + pd.Timedelta(days=6) <= load.index.max() for t in s.index]]
            amin = s.idxmin()
            rows.append({
                "year": y, "search_window": label, "candidate_week": cand.date(),
                "actual_min_week": amin.date(), "offset_weeks": int((amin - cand).days / 7),
                "candidate_load_mw": float(wk[cand]), "min_load_mw": float(s.min()),
                "excess_pct": float((wk[cand] / s.min() - 1) * 100),
                "verdict": "OK" if amin == cand else (
                    "PROCHE (±1 sem.)" if abs((amin - cand).days) <= 7 else "NON VÉRIFIÉE")})
    return pd.DataFrame(rows)


def summer_shape(load: pd.Series, cal_daily: pd.DataFrame, years: list[int], before: int = 4,
                 after: int = 4, anchor_weeks: int = 3) -> pd.DataFrame:
    """Niveau hebdomadaire / niveau d'avant-vacances, par décalage (en semaines) au creux."""
    rows = []
    for y in years:
        trough = sp.summer_trough_week(y)
        anchors = []
        for j in range(before + anchor_weeks, before, -1):
            m = trough - dt.timedelta(weeks=j)
            if common.is_complete(load, [m + dt.timedelta(days=i) for i in range(7)]):
                anchors.append(common.week_mean(load, m))
        if not anchors:
            continue
        a = float(np.mean(anchors))
        for k in range(-before, after + 1):
            m = trough + dt.timedelta(weeks=k)
            if common.is_complete(load, [m + dt.timedelta(days=i) for i in range(7)]):
                rows.append({"year": y, "offset_weeks": k, "ratio": common.week_mean(load, m) / a})
    return pd.DataFrame(rows)


def year_end_table(load: pd.Series, years: list[int]) -> pd.DataFrame:
    """Niveaux S51..S2 rapportés à la baseline S48–S50 et coefficients de Noël / Nouvel An."""
    rows = []
    for y in years:
        try:
            base = common.baseline_level(load, y)
        except ValueError:
            continue
        row = {"year": y, "baseline_mw": base, "xmas_weekday": dt.date(y, 12, 25).strftime("%a")}
        for w in (51, 52, 53):
            if common.has_iso_week(y, w):
                m = common.iso_week_monday(y, w)
                if common.is_complete(load, [m + dt.timedelta(days=i) for i in range(7)]):
                    row[f"S{w}"] = common.week_mean(load, m) / base
        for w in (1, 2, 3):
            m = common.iso_week_monday(y + 1, w)
            if common.is_complete(load, [m + dt.timedelta(days=i) for i in range(7)]):
                row[f"S{w} (Y+1)"] = common.week_mean(load, m) / base
        for lab, d in (("xmas_day", dt.date(y, 12, 25)), ("new_year_day", dt.date(y + 1, 1, 1))):
            if common.day_values(load, d) is not None:
                row[lab] = common.day_mean(load, d) / base
        rows.append(row)
    return pd.DataFrame(rows)


def holiday_table(load: pd.Series, cal_daily: pd.DataFrame) -> pd.DataFrame:
    return ih.holiday_statistics(load, cal_daily).drop(columns=["last_data_date"])


def imputation_check(load: pd.Series, n_trials: int = 40, gap_hours: tuple[int, ...] = (3, 24, 72),
                     seed: int = 0) -> pd.DataFrame:
    """Injecte des trous dans des données réelles et mesure l'erreur de la reconstruction causale."""
    rng = np.random.default_rng(seed)
    rows = []
    for gap in gap_hours:
        errs = []
        for _ in range(n_trials):
            start = int(rng.integers(24 * 35, len(load) - gap - 1))
            s = load.copy()
            truth = s.iloc[start:start + gap].copy()
            s.iloc[start:start + gap] = np.nan
            filled = imputation.impute_causal(s)
            errs.append(np.mean(np.abs(filled.iloc[start:start + gap] - truth) / truth) * 100)
        rows.append({"gap_hours": gap, "mape_pct_mean": float(np.mean(errs)),
                     "mape_pct_p90": float(np.percentile(errs, 90)), "trials": n_trials})
    return pd.DataFrame(rows)
