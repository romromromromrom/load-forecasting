"""Transfert de forme depuis une année analogue (fin d'année S51–S53 / début d'année S1–S2).

Le NIVEAU absolu de l'année analogue n'est jamais réutilisé, seulement des RATIOS :

1. ``baseline`` (niveau normal S48–S50) de l'année cible, nettoyé, connu à l'origine ;
2. ratio semaine analogue / baseline analogue      -> niveau hebdomadaire cible ;
3. coefficient jour/semaine du jour analogue        -> niveau journalier cible ;
4. profil horaire normalisé du jour analogue        -> prévision horaire.

Alignement des semaines : lundi le plus proche de la même position relative à l'ancre
(Noël ou Nouvel An), voir ``map_reference_monday``. Les fériés de l'ancre (25/12 et 1er/01) sont
appariés par IDENTITÉ (25/12 ↔ 25/12) car leur effet dépend du jour, pas de la semaine.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rte_forecast.business_rules import analogue, common
from rte_forecast.business_rules.weather import WeatherNormalizer
from rte_forecast.calendar import special_periods as sp

YEAR_END_METHOD = "YEAR_END_BUSINESS_RULE"
YEAR_START_METHOD = "YEAR_START_BUSINESS_RULE"


@dataclass(frozen=True)
class TransferSpec:
    method: str
    period_type: str
    anchor: str                 # "christmas" | "new_year"
    weeks: tuple[int, ...]
    baseline_year_shift: int    # année ISO des semaines de baseline = iso_year(cible) + shift
    label: str


YEAR_END_SPEC = TransferSpec(YEAR_END_METHOD, sp.YEAR_END, "christmas", (51, 52, 53), 0, "S51–S53")
YEAR_START_SPEC = TransferSpec(YEAR_START_METHOD, sp.YEAR_START, "new_year", (1, 2), -1, "S1–S2")


def period_start(spec: TransferSpec, target_year: int) -> dt.date:
    return common.iso_week_monday(target_year, spec.weeks[0])


def period_days(spec: TransferSpec, target_year: int) -> list[dt.date]:
    return common.iso_weeks_days(target_year, spec.weeks)


def _baseline_ok(load: pd.Series, spec: TransferSpec, year: int) -> bool:
    try:
        common.baseline_level(load, year + spec.baseline_year_shift)
        return True
    except ValueError:
        return False


def available_reference_years(load: pd.Series, spec: TransferSpec, target_year: int) -> list[int]:
    """Années < cible dont la période ET la baseline sont complètes dans l'historique connu."""
    years = []
    for r in range(load.index.min().year, target_year):
        if common.is_complete(load, period_days(spec, r)) and _baseline_ok(load, spec, r):
            years.append(r)
    return years


def _identity_ref_date(d: dt.date, target_year: int, ref_year: int) -> dt.date | None:
    """25/12 et 1er/01 sont appariés par identité de date avec l'année analogue."""
    if (d.month, d.day) in {(12, 25), (1, 1)}:
        return d.replace(year=d.year - (target_year - ref_year))
    return None


def forecast_transfer_period(load: pd.Series, target_year: int, spec: TransferSpec,
                             covid_years=(2020,), normalizer: WeatherNormalizer | None = None,
                             level_mode: str = "planning", recent_lag_days: int = 0
                             ) -> pd.DataFrame:
    """Prévision horaire de toute la période `spec` de `target_year`, émise à son premier jour.

    Retourne un DataFrame indexé par horodate avec `yhat` et les colonnes d'explicabilité.

    ``level_mode`` :
    * ``planning`` (spec) : facteur d'échelle = baseline_cible / baseline_ref (S48–S50), figé à
      l'origine de la période ; aucune donnée postérieure à l'origine n'est lue ;
    * ``adaptive`` : facteur d'échelle ré-ancré chaque jour = charge réelle des 7 jours précédant
      le jour prévu / charge de leurs jours homologues de l'année analogue. Même forme, mêmes
      coefficients ; seul le NIVEAU suit l'information disponible à l'émission. Par défaut
      (``recent_lag_days=0``) on lit les 7 jours avant J (émission à J 00:00) ; avec
      ``recent_lag_days=1`` (émission à J-1 10:00, J-1 incomplet) on lit les 7 jours J-8..J-2.
    """
    if level_mode not in {"planning", "adaptive"}:
        raise ValueError("level_mode doit valoir 'planning' ou 'adaptive'")
    origin = pd.Timestamp(period_start(spec, target_year))
    history = load.loc[: origin - pd.Timedelta(hours=1)]
    g = None
    if normalizer is not None:  # références ramenées à la température normale (voir weather.py)
        history, g = normalizer.adjust(history)
    full_adj = load if normalizer is None else normalizer.adjust_with(load, g)  # tranché par jour

    baseline_year_t = target_year + spec.baseline_year_shift
    baseline_target = common.baseline_level(history, baseline_year_t)

    candidates = available_reference_years(history, spec, target_year)
    ref_year, ranking = analogue.find_best_calendar_analogue(
        target_year, candidates, spec.anchor, preferred_excluded=covid_years)
    row = ranking.loc[ranking["ref_year"] == ref_year].iloc[0]
    baseline_ref = common.baseline_level(history, ref_year + spec.baseline_year_shift)

    a_t = analogue.anchor_date(target_year, spec.anchor)
    a_r = analogue.anchor_date(ref_year, spec.anchor)
    reason = (f"weekday({spec.anchor}) cible={a_t.strftime('%a')} ref={a_r.strftime('%a')}, "
              f"score de similarité={row['score']:.3f}")

    def ref_day_of(x: dt.date) -> dt.date:
        """Jour homologue de `x` dans l'année analogue (même règle que pour les jours prévus)."""
        ident = _identity_ref_date(x, target_year, ref_year)
        if ident is not None:
            return ident
        m_ref = common.map_reference_monday(common.monday_of(x), a_t, a_r)
        return m_ref + dt.timedelta(days=x.weekday())

    def adaptive_scale(day: dt.date) -> tuple[float, float]:
        """(niveau réel récent, niveau homologue récent) sur les 7 jours précédant `day`."""
        cut = pd.Timestamp(day) - pd.Timedelta(days=recent_lag_days) - pd.Timedelta(hours=1)
        hist_d = full_adj.loc[:cut]
        act, ref = [], []
        for j in range(1 + recent_lag_days, 8 + recent_lag_days):
            x = day - dt.timedelta(days=j)
            rx = ref_day_of(x)
            if common.day_values(hist_d, x) is not None and common.day_values(history, rx) is not None:
                act.append(common.day_mean(hist_d, x))
                ref.append(common.day_mean(history, rx))
        if len(act) < 4:
            raise ValueError(f"niveau récent indisponible avant {day}")
        return float(np.mean(act)), float(np.mean(ref))

    rows = []
    days = period_days(spec, target_year)
    day_set = set(days)
    for t_monday in sorted({common.monday_of(d) for d in days}):
        r_monday = common.map_reference_monday(t_monday, a_t, a_r)
        for i in range(7):
            d = t_monday + dt.timedelta(days=i)
            if d not in day_set:
                continue
            r_day = _identity_ref_date(d, target_year, ref_year)
            identity = r_day is not None and common.is_complete(history, [r_day])
            if not identity:
                r_day = r_monday + dt.timedelta(days=i)
            r_week_mon = common.monday_of(r_day)
            week_ref = common.week_mean(history, r_week_mon)
            ratio_week = week_ref / baseline_ref
            coef_day = common.day_coefficient(history, r_day)
            profile = common.normalized_hourly_profile(history, r_day)
            if level_mode == "adaptive":
                recent_act, recent_ref = adaptive_scale(d)
                scale, level_anchor = recent_act / recent_ref, recent_act
            else:
                scale, level_anchor = baseline_target / baseline_ref, baseline_target
            day_level = scale * week_ref * coef_day
            weekly_level_day = scale * week_ref
            for h in range(24):
                rows.append({
                    "timestamp": pd.Timestamp(d) + pd.Timedelta(hours=h),
                    "yhat": day_level * profile[h],
                    "method": spec.method, "reference_year": ref_year, "reference_date": r_day,
                    "reason": reason + (" ; jour apparié par identité (férié d'ancre)"
                                        if identity else ""),
                    "baseline_mw": level_anchor, "baseline_ref_mw": baseline_ref,
                    "weekly_ratio": ratio_week, "weekly_target_level_mw": weekly_level_day,
                    "day_coefficient": coef_day, "day_mean_mw": day_level,
                    "hourly_profile_source": f"{r_day.isoformat()}", "similarity_score": row["score"],
                    "hour": h})
    out = pd.DataFrame(rows).set_index("timestamp")
    if g is not None:
        out["thermosensitivity_heat_mw_per_c"] = g["g_heat"]
        # anomalie de température PRÉVUE de la cible (nulle si prévision = normale)
        out["yhat"] = out["yhat"] + normalizer.forecast_correction(out.index, g)
    out.attrs["ranking"] = ranking
    out.attrs["origin"] = origin
    return out


def recompose_profile(day_mean_value: float, profile: np.ndarray) -> np.ndarray:
    """Recomposition niveau × profil normalisé (utilisée par les tests de cohérence)."""
    return day_mean_value * profile
