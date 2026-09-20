"""Étape 4 : écart de MAE journalière entre deux modèles, avec IC 95 % par bootstrap apparié sur les jours.

    python scripts/04_bootstrap_compare.py --a random_forest_rules_adaptive --b random_forest
    python scripts/04_bootstrap_compare.py --results-dir data/results_issue10h --a xgboost --b sarimax

Écart = MAE journalière(A) - MAE journalière(B), moyennée par période puis ré-échantillonnée avec
remise sur les jours (mêmes jours pour A et B). > 0 : A est MOINS bon que B. Tous les jours
prédits (2022-2024 par défaut) sont utilisés, comme dans RESULTS.md.
"""

import argparse

import numpy as np
import pandas as pd

from rte_forecast.config import load_config, resolve_path
from rte_forecast.evaluation.metrics import PERIOD_LABELS, PERIOD_ORDER


def daily_mae(pred: pd.DataFrame, model: str, mode: str) -> pd.DataFrame:
    p = pred[(pred["model"] == model) & (pred["weather_mode"].isin([mode, "n/a"]))]
    p = p.assign(day=p["timestamp"].dt.normalize(), abs_err=p["error"].abs())
    g = p.groupby("day").agg(mae=("abs_err", "mean"), period_type=("period_type", "first"))
    g["period"] = g["period_type"].map(PERIOD_LABELS)
    return g


def bootstrap_diff(d: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), *(float(x) for x in np.percentile(means, [2.5, 97.5]))


def compare(pred: pd.DataFrame, a: str, b: str, mode: str, n_boot: int = 4000, seed: int = 0
            ) -> pd.DataFrame:
    da, db = daily_mae(pred, a, mode), daily_mae(pred, b, mode)
    both = da.join(db, lsuffix="_a", rsuffix="_b", how="inner")
    both["diff"] = both["mae_a"] - both["mae_b"]
    rng = np.random.default_rng(seed)
    rows = []
    for period in [*PERIOD_ORDER[:-1], "ALL"]:
        sub = both if period == "ALL" else both[both["period_a"] == period]
        if len(sub):
            m, lo, hi = bootstrap_diff(sub["diff"].to_numpy(), n_boot, rng)
            rows.append({"period": period, "n_days": len(sub), "diff_mw": m, "ci_low": lo,
                         "ci_high": hi, "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--weather", nargs="+", default=["normal", "noisy"])
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = resolve_path(load_config(), "results_dir") if a.results_dir is None else \
        __import__("pathlib").Path(a.results_dir)
    pred = pd.read_parquet(res / "predictions.parquet")
    for mode in a.weather:
        t = compare(pred, a.a, a.b, mode, a.n_boot, a.seed)
        t["IC 95 %"] = t.apply(lambda r: f"{r.diff_mw:+,.0f} [{r.ci_low:+,.0f} ; {r.ci_high:+,.0f}]"
                               + (" *" if r.significant else ""), axis=1)
        print(f"\n== {a.a} - {a.b} | météo '{mode}' | {res.name} (* = IC exclut 0)")
        print(t[["period", "n_days", "IC 95 %"]].to_string(index=False))
