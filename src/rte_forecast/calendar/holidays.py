"""Jours fériés France métropolitaine (bibliothèque `holidays`, sans Alsace-Moselle)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import holidays as _holidays


def french_holidays(years: Iterable[int]) -> dict[dt.date, str]:
    """{date: nom} des jours fériés légaux nationaux pour les années demandées."""
    years = sorted(set(int(y) for y in years))
    return {d: n for d, n in sorted(_holidays.France(years=years).items())}


def christmas(year: int) -> dt.date:
    return dt.date(year, 12, 25)


def new_year(year: int) -> dt.date:
    return dt.date(year, 1, 1)
