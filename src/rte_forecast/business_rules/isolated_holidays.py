"""Jours fériés « isolés » et ponts (hors fin/début d'année ; l'été gère le 15 août lui-même).

Principe (spec §11), sans règle inventée : les coefficients sont MESURÉS sur l'historique.

1. niveau attendu = prévision du modèle « normal » pour ce jour, comme s'il était ordinaire ;
2. occurrences historiques comparables, par ordre de préférence :
     T1  même férié ET même classe de jour de semaine (lun / mar-jeu / ven / sam / dim) ;
     T2  n'importe quel férié de même classe de jour de semaine ;
     T3  même férié, toutes classes ;
   (ponts : T1 = même jour de semaine, T2 = tous les ponts) ;
3. ratio historique = charge du jour férié / charge moyenne du MÊME jour de semaine
   à ±7 jours (repli ±14 jours) parmi les jours ordinaires ;
4. coefficient = médiane des ratios retenus ; profil horaire = médiane des profils normalisés ;
5. prévision = niveau attendu x coefficient x profil normalisé.

Une occurrence n'est utilisable que si TOUTES les données qu'elle a consommées (jour + voisins)
précèdent le jour prévu : c'est la protection anti-fuite propre à ce module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rte_forecast.business_rules import common
from rte_forecast.calendar import special_periods as sp

ISOLATED_METHOD = "ISOLATED_HOLIDAY_BUSINESS_RULE"
BRIDGE_METHOD = "BRIDGE_DAY_BUSINESS_RULE"
WEEKDAY_CLASS = {0: "MON", 1: "MIDWEEK", 2: "MIDWEEK", 3: "MIDWEEK", 4: "FRI", 5: "SAT", 6: "SUN"}
EXCLUDED_NEIGHBOR_TYPES = {sp.YEAR_END, sp.YEAR_START}
NEIGHBOR_OFFSETS = ((-7, 7), (-14, 14))


def _is_ordinary(cal_daily: pd.DataFrame, d: pd.Timestamp) -> bool:
    if d not in cal_daily.index:
        return False
    r = cal_daily.loc[d]
    return not (r["is_public_holiday"] or r["is_bridge_day"]
                or r["special_period_type"] in EXCLUDED_NEIGHBOR_TYPES)


def holiday_statistics(load: pd.Series, cal_daily: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par férié/pont observé : ratio à la charge « ordinaire » environnante."""
    rows = []
    for ts, r in cal_daily[(cal_daily["is_public_holiday"] | cal_daily["is_bridge_day"])
                           & ~cal_daily["special_period_type"].isin(EXCLUDED_NEIGHBOR_TYPES)
                           ].iterrows():
        day = ts.date()
        if common.day_values(load, day) is None:
            continue
        used = []
        for pair in NEIGHBOR_OFFSETS:
            used = [ts + pd.Timedelta(days=o) for o in pair
                    if _is_ordinary(cal_daily, ts + pd.Timedelta(days=o))
                    and common.day_values(load, (ts + pd.Timedelta(days=o)).date()) is not None]
            if used:
                break
        if not used:
            continue
        surround = float(np.mean([common.day_mean(load, u.date()) for u in used]))
        kind = "holiday" if r["is_public_holiday"] else "bridge"
        rows.append({"date": ts, "kind": kind,
                     "name": r["holiday_name"] if kind == "holiday" else "PONT",
                     "weekday": int(r["weekday"]), "weekday_class": WEEKDAY_CLASS[int(r["weekday"])],
                     "period_type": r["special_period_type"], "day_mean_mw": common.day_mean(load, day),
                     "surrounding_mw": surround, "ratio": common.day_mean(load, day) / surround,
                     "last_data_date": max(used + [ts])})
    return pd.DataFrame(rows)


def select_occurrences(stats: pd.DataFrame, target: pd.Timestamp, kind: str, name: str,
                       weekday: int) -> tuple[pd.DataFrame, str]:
    """Occurrences comparables disponibles STRICTEMENT avant `target` et niveau de repli retenu."""
    s = stats[(stats["kind"] == kind) & (stats["last_data_date"] < target)]
    cls = WEEKDAY_CLASS[weekday]
    if kind == "bridge":
        tiers = [("T1 même jour de semaine", s[s["weekday"] == weekday]), ("T2 tous ponts", s)]
    else:
        tiers = [("T1 même férié + même classe de jour", s[(s["name"] == name)
                                                          & (s["weekday_class"] == cls)]),
                 ("T2 même classe de jour, tous fériés", s[s["weekday_class"] == cls]),
                 ("T3 même férié, toutes classes", s[s["name"] == name])]
    for label, sel in tiers:
        if len(sel):
            return sel, label
    return s.iloc[0:0], "AUCUNE"


def forecast_isolated_holiday(target: pd.Timestamp, normal_hourly: np.ndarray, stats: pd.DataFrame,
                              history: pd.Series, cal_row: pd.Series) -> dict | None:
    """Prévision d'un férié/pont ; None si aucune occurrence historique exploitable.

    `normal_hourly` : 24 valeurs du modèle « normal » pour ce jour. `history` ne doit contenir
    que des données antérieures à `target`.
    """
    kind = "holiday" if cal_row["is_public_holiday"] else "bridge"
    name = cal_row["holiday_name"] if kind == "holiday" else "PONT"
    occ, tier = select_occurrences(stats, target, kind, name, int(cal_row["weekday"]))
    if occ.empty:
        return None
    profiles = np.array([common.normalized_hourly_profile(history, d.date()) for d in occ["date"]])
    profile = np.median(profiles, axis=0)
    profile = profile / profile.mean()
    coef = float(np.median(occ["ratio"]))
    level = float(np.mean(normal_hourly))
    return {
        "yhat": level * coef * profile, "method": ISOLATED_METHOD if kind == "holiday"
        else BRIDGE_METHOD, "reference_year": ",".join(str(d.year) for d in occ["date"]),
        "reason": f"{tier} ; n={len(occ)} ; ratios={[round(x, 3) for x in occ['ratio']]}",
        "expected_normal_level_mw": level, "day_coefficient": coef,
        "day_mean_mw": level * coef,
        "hourly_profile_source": "médiane de " + ", ".join(d.strftime("%Y-%m-%d") for d in occ["date"]),
        "n_occurrences": len(occ)}


def stats_ready(load: pd.Series, cal_daily: pd.DataFrame) -> pd.DataFrame:
    """Alias lisible : statistiques historiques calculées une fois (filtrées ensuite par date)."""
    return holiday_statistics(load, cal_daily)
