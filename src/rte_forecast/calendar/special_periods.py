"""Périodes calendaires spéciales et classification journalière/horaire.

Périodes (déterminées UNIQUEMENT à partir du calendrier, donc connues à l'avance, sans fuite) :

* YEAR_END   : semaines ISO 51, 52, 53 (ramp-down de fin d'année)      -> ``is_ramp_down``
* YEAR_START : semaines ISO 1, 2 (ramp-up de janvier)                  -> ``is_ramp_up``
* SUMMER     : fenêtre de ± N semaines autour de la « semaine du creux estival », la semaine ISO
               dont le week-end est le plus proche du 15 août       -> ``is_august_special_period``
* HOLIDAY    : jour férié hors périodes ci-dessus
* BRIDGE     : jour de pont hors périodes ci-dessus
* NORMAL     : tout le reste (seul régime appris par le modèle « normal »)

Précédence : YEAR_END > YEAR_START > SUMMER > HOLIDAY > BRIDGE > NORMAL. Les drapeaux
``is_public_holiday`` / ``is_bridge_day`` restent vrais même lorsque le type retenu est une
période spéciale.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from rte_forecast.calendar.bridge_days import find_bridge_days
from rte_forecast.calendar.holidays import french_holidays

YEAR_END, YEAR_START, SUMMER = "YEAR_END", "YEAR_START", "SUMMER"
HOLIDAY, BRIDGE, NORMAL = "HOLIDAY", "BRIDGE", "NORMAL"
PERIOD_ORDER = [NORMAL, HOLIDAY, BRIDGE, SUMMER, YEAR_END, YEAR_START]

DEFAULT_YEAR_END_WEEKS = (51, 52, 53)
DEFAULT_YEAR_START_WEEKS = (1, 2)


def summer_trough_week(year: int) -> dt.date:
    """Lundi de la semaine ISO dont le week-end est le plus proche du 15 août.

    Distance = min(|samedi - 15/08|, |dimanche - 15/08|). Égalité (15/08 un mercredi) : on
    retient le week-end POSTÉRIEUR, règle arbitraire mais déterministe et documentée.
    """
    aug15 = dt.date(year, 8, 15)
    best: tuple[int, int, dt.date] | None = None
    for offset in range(-10, 11):
        sat = aug15 + dt.timedelta(days=offset)
        if sat.weekday() != 5:
            continue
        dist = min(abs((sat - aug15).days), abs((sat + dt.timedelta(days=1) - aug15).days))
        key = (dist, -(sat.toordinal()))  # égalité -> week-end le plus tardif
        if best is None or key < best[:2]:
            best = (key[0], key[1], sat)
    assert best is not None
    return best[2] - dt.timedelta(days=5)  # samedi -> lundi


def summer_window(year: int, weeks_before: int = 2, weeks_after: int = 2) -> tuple[dt.date, dt.date]:
    """(premier jour, dernier jour) de la fenêtre estivale spéciale de `year`."""
    monday = summer_trough_week(year)
    return (monday - dt.timedelta(weeks=weeks_before),
            monday + dt.timedelta(weeks=weeks_after, days=6))


def build_daily_calendar(start: str | dt.date, end: str | dt.date, *,
                         year_end_weeks=DEFAULT_YEAR_END_WEEKS,
                         year_start_weeks=DEFAULT_YEAR_START_WEEKS,
                         summer_weeks_before: int = 2, summer_weeks_after: int = 2) -> pd.DataFrame:
    """Calendrier journalier classifié (index = date normalisée)."""
    days = pd.date_range(start, end, freq="D")
    # marge d'un an pour que les fériés/ponts voisins des bornes soient corrects
    hol = french_holidays(range(days.year.min() - 1, days.year.max() + 2))
    hol_dates = set(hol)
    all_days = pd.date_range(days.min() - pd.Timedelta(days=2), days.max() + pd.Timedelta(days=2))
    bridges = find_bridge_days((d.date() for d in all_days), hol_dates)

    iso = days.isocalendar()
    cal = pd.DataFrame(index=days)
    cal["year"] = days.year
    cal["month"] = days.month
    cal["iso_year"] = iso["year"].to_numpy()
    cal["iso_week"] = iso["week"].to_numpy()
    cal["weekday"] = days.weekday
    cal["is_weekend"] = cal["weekday"] >= 5
    cal["is_public_holiday"] = [d.date() in hol_dates for d in days]
    cal["holiday_name"] = [hol.get(d.date(), "") for d in days]
    cal["is_bridge_day"] = [d.date() in bridges for d in days]
    cal["is_ramp_down"] = cal["iso_week"].isin(year_end_weeks)
    cal["is_ramp_up"] = cal["iso_week"].isin(year_start_weeks)

    in_summer = np.zeros(len(days), dtype=bool)
    for y in range(days.year.min(), days.year.max() + 1):
        s, e = summer_window(y, summer_weeks_before, summer_weeks_after)
        in_summer |= (days >= pd.Timestamp(s)) & (days <= pd.Timestamp(e))
    cal["is_august_special_period"] = in_summer

    ptype = np.full(len(days), NORMAL, dtype=object)
    ptype[cal["is_bridge_day"].to_numpy()] = BRIDGE
    ptype[cal["is_public_holiday"].to_numpy()] = HOLIDAY
    ptype[cal["is_august_special_period"].to_numpy()] = SUMMER
    ptype[cal["is_ramp_up"].to_numpy()] = YEAR_START
    ptype[cal["is_ramp_down"].to_numpy()] = YEAR_END
    cal["special_period_type"] = ptype
    cal["is_normal_day"] = cal["special_period_type"] == NORMAL
    cal.index.name = "date"
    return cal


def build_hourly_calendar(index: pd.DatetimeIndex, **kwargs) -> pd.DataFrame:
    """Calendrier au pas horaire (jour broadcasté + colonne `hour`)."""
    daily = build_daily_calendar(index.min().normalize(), index.max().normalize(), **kwargs)
    cal = daily.reindex(index.normalize())
    cal.index = index
    cal.index.name = "timestamp"
    cal.insert(0, "hour", index.hour)
    return cal
