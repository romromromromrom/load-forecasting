"""Briques communes des moteurs de règles : niveaux, coefficients, profils normalisés.

Chaîne hiérarchique (visible ici et dans chaque moteur) :

    niveau de référence  ->  niveau hebdomadaire  ->  coefficient jour/semaine
                         ->  profil horaire normalisé  ->  prévision horaire

    prévision(d, h) = niveau_ref_cible x ratio_semaine x coef_jour x profil_horaire(h)
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import numpy as np
import pandas as pd

from rte_forecast.data.cleaning import detect_invalid
from rte_forecast.data.imputation import impute_causal

DAY_H = 24
WEEK_H = 168


def iso_week_monday(iso_year: int, iso_week: int) -> dt.date:
    return dt.date.fromisocalendar(iso_year, iso_week, 1)


def has_iso_week(iso_year: int, iso_week: int) -> bool:
    """Vrai si la semaine ISO existe (la semaine 53 n'existe que certaines années)."""
    if iso_week <= 52:
        return True
    return dt.date(iso_year, 12, 28).isocalendar()[1] == 53


def iso_weeks_days(iso_year: int, weeks: Iterable[int]) -> list[dt.date]:
    """Tous les jours (lundi→dimanche) des semaines ISO demandées qui existent."""
    days: list[dt.date] = []
    for w in weeks:
        if has_iso_week(iso_year, w):
            mon = iso_week_monday(iso_year, w)
            days.extend(mon + dt.timedelta(days=i) for i in range(7))
    return days


def monday_of(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def day_values(load: pd.Series, day: dt.date) -> np.ndarray | None:
    """24 valeurs horaires du jour, ou None si le jour est incomplet/absent."""
    t = pd.Timestamp(day)
    seg = load.loc[t: t + pd.Timedelta(hours=23)]
    if len(seg) != DAY_H or seg.isna().any():
        return None
    return seg.to_numpy(dtype=float)


def is_complete(load: pd.Series, days: Iterable[dt.date]) -> bool:
    return all(day_values(load, d) is not None for d in days)


def day_mean(load: pd.Series, day: dt.date) -> float:
    v = day_values(load, day)
    if v is None:
        raise ValueError(f"jour incomplet ou absent : {day}")
    return float(v.mean())


def week_mean(load: pd.Series, monday: dt.date) -> float:
    """Niveau hebdomadaire = moyenne des 168 heures de la semaine lundi→dimanche."""
    vals = [day_values(load, monday + dt.timedelta(days=i)) for i in range(7)]
    if any(v is None for v in vals):
        raise ValueError(f"semaine incomplète : {monday}")
    return float(np.mean(np.concatenate(vals)))


def day_coefficient(load: pd.Series, day: dt.date) -> float:
    """Coefficient journalier = moyenne du jour / moyenne de SA semaine (lundi→dimanche)."""
    return day_mean(load, day) / week_mean(load, monday_of(day))


def normalized_hourly_profile(load: pd.Series, day: dt.date) -> np.ndarray:
    """Profil intraday normalisé : load(h) / moyenne(jour). Sa moyenne vaut exactement 1."""
    v = day_values(load, day)
    if v is None:
        raise ValueError(f"jour incomplet ou absent : {day}")
    return v / v.mean()


def clean_reference_window(load: pd.Series, start: dt.date, end: dt.date) -> pd.Series:
    """Nettoie une fenêtre de référence AVANT tout calcul de niveau.

    Détecte manquants/valeurs invalides puis reconstruit CAUSALEMENT (aucune donnée postérieure
    à l'instant reconstruit). On charge 4 semaines d'historique préalable pour les références.
    """
    t0 = pd.Timestamp(start) - pd.Timedelta(weeks=4)
    t1 = pd.Timestamp(end) + pd.Timedelta(hours=23)
    seg = load.loc[t0:t1]
    if len(seg) == 0:
        return seg
    seg = seg.reindex(pd.date_range(seg.index.min(), t1, freq="h"))
    invalid = detect_invalid(seg)
    return impute_causal(seg, invalid).loc[pd.Timestamp(start):t1]


def baseline_level(load: pd.Series, iso_year: int, weeks: Iterable[int] = (48, 49, 50)) -> float:
    """Niveau normal moyen (MW) sur les semaines ISO de référence (défaut S48–S50).

    ``baseline_pre_christmas`` = moyenne horaire des semaines S48–S50 de `iso_year` après
    nettoyage/rebouchage causal. Lève ValueError si la fenêtre n'existe pas dans `load`.
    """
    days = iso_weeks_days(iso_year, weeks)
    seg = clean_reference_window(load, days[0], days[-1])
    if len(seg) != len(days) * DAY_H or seg.isna().any():
        raise ValueError(f"fenêtre de référence indisponible : {iso_year} S{list(weeks)}")
    return float(seg.mean())


def map_reference_monday(target_monday: dt.date, anchor_target: dt.date,
                         anchor_ref: dt.date) -> dt.date:
    """Lundi de l'année de référence dont la position vs l'ancre est la plus proche de la cible.

    On reporte le décalage (en jours) du lundi cible par rapport à l'ancre (Noël ou Nouvel An)
    sur l'année de référence, puis on choisit le LUNDI le plus proche : même jour de semaine,
    écart de position d'au plus 3 jours. Si la semaine de l'ancre coïncide, l'écart est nul.
    """
    guess = anchor_ref + dt.timedelta(days=(target_monday - anchor_target).days)
    w = guess.weekday()
    return guess - dt.timedelta(days=w) if w <= 3 else guess + dt.timedelta(days=7 - w)
