"""Jours de pont.

Définition (spec) : jour OUVRÉ situé directement entre un jour férié et un week-end, ou entre
un week-end et un jour férié :

    jeudi férié + vendredi ouvré + samedi        -> vendredi = pont
    dimanche + lundi ouvré + mardi férié          -> lundi = pont

Un férié en milieu de semaine (mercredi) ne crée donc AUCUN pont selon cette définition
stricte ; c'est un choix assumé et documenté, pas un oubli.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable


def _is_weekend(d: dt.date) -> bool:
    return d.weekday() >= 5


def is_bridge_day(d: dt.date, holiday_dates: set[dt.date]) -> bool:
    """Vrai si `d` est un jour ouvré non férié coincé entre férié et week-end."""
    if _is_weekend(d) or d in holiday_dates:
        return False
    before, after = d - dt.timedelta(days=1), d + dt.timedelta(days=1)
    before_off_hol = before in holiday_dates
    after_off_hol = after in holiday_dates
    return (_is_weekend(before) and after_off_hol) or (before_off_hol and _is_weekend(after))


def find_bridge_days(days: Iterable[dt.date], holiday_dates: set[dt.date]) -> set[dt.date]:
    return {d for d in days if is_bridge_day(d, holiday_dates)}
