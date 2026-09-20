"""Recherche d'une année historique analogue (calendrier de fin/début d'année).

Score de similarité calendaire, dans [0, 1] (1 = calendriers identiques autour de l'ancre) :

    score = 1 - ( 0.50 * d_weekday / 3          jour de semaine de l'ancre (distance circulaire)
                + 0.20 * min(d_isoweek, 2) / 2  position de l'ancre dans les semaines ISO
                + 0.30 * d_structure )          structure W/H/B sur la fenêtre [-7, +14] jours

* ``anchor`` = Noël (25/12) pour la fin d'année, Nouvel An (1/01) pour le redémarrage.
* ``d_structure`` = taux de désaccord jour par jour du type de journée (W week-end, H férié,
  B pont, O ouvré ordinaire).
* Départage des égalités : année la plus RÉCENTE (déterministe, donc reproductible).

Noël et le Nouvel An tombent toujours le même jour de semaine (7 jours d'écart) ; la distinction
Noël/Nouvel An ne joue que via la fenêtre de structure et la semaine ISO.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd

from rte_forecast.calendar.bridge_days import is_bridge_day
from rte_forecast.calendar.holidays import french_holidays

STRUCT_OFFSETS = range(-7, 15)
W_WEEKDAY, W_ISOWEEK, W_STRUCT = 0.50, 0.20, 0.30


def anchor_date(year: int, anchor: str) -> dt.date:
    if anchor == "christmas":
        return dt.date(year, 12, 25)
    if anchor == "new_year":
        return dt.date(year, 1, 1)
    raise ValueError("anchor doit valoir 'christmas' ou 'new_year'")


def _iso_position(d: dt.date) -> int:
    """Rang de semaine croissant sur la jonction d'années (S1 est placée après S52/S53)."""
    y, w, _ = d.isocalendar()
    return w if w >= 40 else w + 53


def day_type_string(anchor_d: dt.date) -> str:
    """Structure W/H/B/O de [anchor-7, anchor+14] jours."""
    hol = set(french_holidays(range(anchor_d.year - 1, anchor_d.year + 2)))
    out = []
    for off in STRUCT_OFFSETS:
        d = anchor_d + dt.timedelta(days=off)
        if d in hol:
            out.append("H")
        elif d.weekday() >= 5:
            out.append("W")
        elif is_bridge_day(d, hol):
            out.append("B")
        else:
            out.append("O")
    return "".join(out)


def calendar_similarity(target_year: int, ref_year: int, anchor: str = "christmas") -> dict:
    """Composantes et score de similarité entre deux calendriers de fin/début d'année."""
    at, ar = anchor_date(target_year, anchor), anchor_date(ref_year, anchor)
    raw = abs(at.weekday() - ar.weekday())
    d_weekday = min(raw, 7 - raw)
    d_iso = min(abs(_iso_position(at) - _iso_position(ar)), 2)
    st, sr = day_type_string(at), day_type_string(ar)
    d_struct = sum(a != b for a, b in zip(st, sr, strict=True)) / len(st)
    score = 1 - (W_WEEKDAY * d_weekday / 3 + W_ISOWEEK * d_iso / 2 + W_STRUCT * d_struct)
    return {"ref_year": ref_year, "score": round(score, 6), "d_weekday": d_weekday,
            "d_isoweek": d_iso, "d_structure": round(d_struct, 4), "weekday_target": at.weekday(),
            "weekday_ref": ar.weekday()}


def rank_analogues(target_year: int, candidate_years: Iterable[int], anchor: str = "christmas"
                   ) -> pd.DataFrame:
    """Classement complet (score décroissant, puis année la plus récente)."""
    rows = [calendar_similarity(target_year, y, anchor) for y in candidate_years if y != target_year]
    if not rows:
        return pd.DataFrame(columns=["ref_year", "score"])
    return (pd.DataFrame(rows).sort_values(["score", "ref_year"], ascending=[False, False])
            .reset_index(drop=True))


def find_best_calendar_analogue(target_year: int, candidate_years: Iterable[int],
                                anchor: str = "christmas",
                                preferred_excluded: Iterable[int] = ()) -> tuple[int, pd.DataFrame]:
    """Meilleure année analogue et tableau de classement.

    ``preferred_excluded`` (ex. années COVID) est écarté tant qu'il reste d'autres candidats ;
    sinon on le réintègre (le choix est alors marqué dans le tableau via `excluded`).
    """
    candidates = sorted(set(candidate_years))
    excl = set(preferred_excluded)
    pool = [y for y in candidates if y not in excl] or candidates
    ranking = rank_analogues(target_year, candidates, anchor)
    ranking["excluded"] = ranking["ref_year"].isin(excl)
    usable = ranking[ranking["ref_year"].isin(pool)]
    if usable.empty:
        raise ValueError(f"aucune année analogue disponible pour {target_year}")
    return int(usable.iloc[0]["ref_year"]), ranking


def find_best_year_end_calendar_analogue(target_year: int,
                                         candidate_years: Iterable[int] | None = None,
                                         preferred_excluded: Iterable[int] = ()
                                         ) -> tuple[int, pd.DataFrame]:
    """Année analogue pour Noël/fin d'année (ancre = 25/12).

    Sans `candidate_years`, on considère les 10 années précédentes.
    """
    candidate_years = range(target_year - 10, target_year) if candidate_years is None \
        else candidate_years
    return find_best_calendar_analogue(target_year, candidate_years, "christmas", preferred_excluded)


def find_best_year_start_calendar_analogue(target_year: int,
                                           candidate_years: Iterable[int] | None = None,
                                           preferred_excluded: Iterable[int] = ()
                                           ) -> tuple[int, pd.DataFrame]:
    """Année analogue pour le redémarrage de janvier (ancre = 1er janvier)."""
    candidate_years = range(target_year - 10, target_year) if candidate_years is None \
        else candidate_years
    return find_best_calendar_analogue(target_year, candidate_years, "new_year", preferred_excluded)
