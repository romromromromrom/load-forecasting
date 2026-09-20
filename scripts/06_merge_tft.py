"""Étape 6 : fusionne predictions_tft.parquet (calculé sur GPU) dans un dossier de résultats du benchmark.

    python scripts/06_merge_tft.py --tft-dir ~/Downloads/tft_results --results-dir data/results_issue10h

Garde-fous : le TFT doit avoir été entraîné avec le MÊME instant d'émission que le benchmark cible
(sinon la comparaison serait faussée), et sa cible `y_true` doit être identique. Idempotent : les
lignes `tft` déjà présentes sont remplacées. `predictions.parquet` est sauvegardé une fois sous
`predictions_without_tft.parquet`. Les tables de métriques sont recalculées comme dans run_all.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from rte_forecast.evaluation import metrics

LABEL = "TFT (compact, quantiles)"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tft-dir", required=True)
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--force", action="store_true", help="ignore le contrôle d'instant d'émission")
    a = ap.parse_args()
    res, tft_dir = Path(a.results_dir), Path(a.tft_dir)

    meta = json.loads((res / "run_meta.json").read_text())
    tft = pd.read_parquet(tft_dir / "predictions_tft.parquet")
    fold_stats = pd.read_parquet(tft_dir / "tft_fold_stats.parquet")

    bench_issue = -1 if meta.get("issue_hour") is None else int(meta["issue_hour"])
    tft_issues = {int(x) for x in fold_stats["issue_hour"].unique()}
    if tft_issues != {bench_issue} and not a.force:
        raise SystemExit(f"instant d'émission différent : benchmark={bench_issue} (-1 = D 00:00), "
                         f"TFT={sorted(tft_issues)}. Réentraînez le TFT avec --issue-hour ou --force.")

    backup = res / "predictions_without_tft.parquet"
    if not backup.exists():
        shutil.copy(res / "predictions.parquet", backup)
    base = pd.read_parquet(backup)
    base = base[base["model"] != "tft"]

    # cible identique à celle du benchmark (mêmes heures, mêmes valeurs)
    ref = base[base["model"] == "naive_j7"].set_index("timestamp")["y_true"]
    chk = tft.set_index("timestamp")["y_true"]
    common = chk.index.intersection(ref.index)
    if len(common) != chk.index.nunique() or not np.allclose(chk.groupby(level=0).first().loc[common],
                                                            ref.loc[common]):
        raise SystemExit("y_true du TFT différent de celui du benchmark : données ou folds incohérents")

    allp = pd.concat([base, tft], ignore_index=True)
    allp.to_parquet(res / "predictions.parquet")

    key = ["model", "weather_mode"]
    metrics.metrics_by_period(allp, key + ["split"]).to_parquet(res / "metrics_split.parquet")
    metrics.metrics_by_period(allp, key).to_parquet(res / "metrics_period.parquet")
    old = pd.read_parquet(res / "cv_folds.parquet")
    old = old[old["model"] != "tft"]
    fm = (metrics.metrics_by_period(tft, key + ["fold_id"]).query("period == 'ALL'")
          .drop(columns="period"))
    fm = fm.merge(fold_stats[key + ["fold_id", "fold_label", "fit_seconds", "n_train"]],
                  on=key + ["fold_id"], how="left")
    pd.concat([old, fm], ignore_index=True).to_parquet(res / "cv_folds.parquet")

    meta["models"]["tft"] = LABEL
    meta["tft"] = {"weather_modes": sorted(tft["weather_mode"].unique().tolist()),
                   "folds": sorted(int(x) for x in tft["fold_id"].unique()),
                   "seeds": int(fold_stats["seeds"].max())}
    (res / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"TFT fusionné dans {res} ({len(tft):,} lignes). MAE par scénario / split :")
    print(metrics.metrics_by_period(tft, key + ["split"]).query("period == 'ALL'")
          [key + ["split", "mae", "rmse", "bias"]].round(0).to_string(index=False))
