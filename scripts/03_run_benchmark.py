"""Étape 3 : benchmark complet (hyperparamètres -> walk-forward -> métriques -> SHAP)."""

import argparse

from rte_forecast.config import load_config
from rte_forecast.evaluation.backtest import run_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", nargs="+", default=["normal", "noisy", "perfect"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="smoke test : 2 folds, grille réduite")
    ap.add_argument("--skip-sarimax", action="store_true")
    ap.add_argument("--issue-hour", type=int, default=None,
                    help="dernière heure de D-1 connue à l'émission (9 = émission à D-1 10:00) ; "
                         "défaut : émission à D 00:00")
    ap.add_argument("--save-models", action="store_true",
                    help="sauvegarde chaque modèle entraîné (par fold + modèle final) dans --models-dir")
    ap.add_argument("--models-dir", default=None, help="défaut : paths.models_dir de la config")
    ap.add_argument("--results-dir", default=None,
                    help="dossier de sortie (à fixer pour ne pas écraser data/results)")
    a = ap.parse_args()
    cfg = load_config()
    if a.issue_hour is not None:
        cfg["features"]["issue_hour"] = a.issue_hour
    if a.results_dir:
        cfg["paths"]["results_dir"] = a.results_dir
    run_all(cfg, weather_modes=a.weather, jobs=a.jobs, quick=a.quick, skip_sarimax=a.skip_sarimax,
            save_models=a.save_models, models_dir=a.models_dir)
