"""Étape 5 : backtest walk-forward du Temporal Fusion Transformer (à lancer sur GPU : Colab/Kaggle).

Exemples
    python scripts/05_train_tft.py --quick                                    # test de fumée (1 fold, 2 epochs)
    python scripts/05_train_tft.py --weather normal --issue-hour 9 --out-dir /content/drive/MyDrive/rte/tft
    python scripts/05_train_tft.py --weather normal noisy --issue-hour 9 --seeds 3 --amp --resume ...

Sortie (dans --out-dir) : predictions_tft.parquet (même schéma que data/results/predictions.parquet
+ colonnes q10/q90), tft_fold_stats.parquet, weights/*.pt. Fusion locale : scripts/06_merge_tft.py.
"""

import argparse
from dataclasses import fields

from rte_forecast.evaluation.tft_backtest import TFTConfig, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weather", nargs="+", default=["normal"], choices=["normal", "noisy", "perfect"])
    ap.add_argument("--issue-hour", type=int, default=None,
                    help="dernière heure de D-1 connue à l'émission (9 = émission à D-1 10:00) ; "
                         "défaut : émission à D 00:00. À aligner sur le benchmark comparé !")
    ap.add_argument("--out-dir", default="tft_results")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="numéros de folds (défaut : 1..12)")
    ap.add_argument("--resume", action="store_true", help="ignore les (météo, fold) déjà écrits")
    ap.add_argument("--quick", action="store_true", help="test de fumée : dernier fold, 2 epochs")
    ap.add_argument("--device", default=None, help="cuda | cpu (défaut : auto)")
    d = TFTConfig()
    ap.add_argument("--lookback", type=int, default=d.lookback, help="heures de contexte passé")
    ap.add_argument("--d-model", type=int, default=d.d_model)
    ap.add_argument("--heads", type=int, default=d.n_heads)
    ap.add_argument("--lstm-layers", type=int, default=d.lstm_layers)
    ap.add_argument("--dropout", type=float, default=d.dropout)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--val-days", type=int, default=d.val_days)
    ap.add_argument("--seeds", type=int, default=d.seeds, help="ensemble de N graines (moyenne)")
    ap.add_argument("--amp", action="store_true", help="précision mixte fp16 (GPU)")
    a = ap.parse_args()

    known = {f.name for f in fields(TFTConfig)}
    kw = {"lookback": a.lookback, "d_model": a.d_model, "n_heads": a.heads,
          "lstm_layers": a.lstm_layers, "dropout": a.dropout, "lr": a.lr, "batch_size": a.batch_size,
          "epochs": a.epochs, "patience": a.patience, "val_days": a.val_days, "seeds": a.seeds,
          "amp": a.amp}
    assert set(kw) <= known
    run(TFTConfig(**kw), weather_modes=a.weather, issue_hour=a.issue_hour, out_dir=a.out_dir,
        folds_subset=a.folds, resume=a.resume, quick=a.quick, device=a.device)
