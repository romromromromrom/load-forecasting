# rte-load-forecasting-demo

Démonstrateur end-to-end de **prévision de la consommation électrique française** (RTE eCO2mix +
température Enedis), orienté production/MLOps : pipeline simple, correct, reproductible et testé.
Pas d'Airflow / Docker / MLflow / PyTorch à ce stade, volontairement.

> Le modèle le plus simple capable de battre sérieusement la baseline vient d'abord ; un modèle plus
> complexe n'est justifié que si son gain hors échantillon paie sa complexité opérationnelle.

## Idée directrice

* Un modèle d'arbres (**XGBoost / Random Forest**) apprend le **régime normal** de consommation
  (jours ouvrés/week-ends ordinaires) ;
* les régimes calendaires rares (fin d'année S51–S53, redémarrage S1–S2, creux du 15 août, fériés,
  ponts) sont **exclus de son apprentissage** et reconstruits par un **moteur de règles métier**
  explicable, à partir d'années analogues (niveau / semaine / jour / profil horaire) ;
* un **routeur** (`forecast.py`) choisit le moteur, un **backtest walk-forward** compare hybride et
  modèle seul, période par période.

## Démarrage

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev,dashboard]"   # ou pip
python scripts/01_prepare_data.py          # eCO2mix + Enedis (+ proxy Open-Meteo) -> Parquet
python scripts/02_analyze_hypotheses.py    # vérification empirique des hypothèses métier
python scripts/03_run_benchmark.py         # hyperparamètres, walk-forward, SHAP (~20 min)
streamlit run app/dashboard.py             # comparer les modèles, heatmaps d'erreurs, etc.
python scripts/03_run_benchmark.py --issue-hour 9 --results-dir data/results_issue10h  # émission à D-1 10:00
# TFT (GPU) : notebooks/colab_tft.ipynb -> scripts/05_train_tft.py, puis en local :
python scripts/04_bootstrap_compare.py --results-dir data/results_issue10h --a random_forest_rules_adaptive --b random_forest  # IC bootstrap
python scripts/06_merge_tft.py --tft-dir <dossier_téléchargé> --results-dir data/results_issue10h
pytest && ruff check .
```

## Arborescence

```
configs/default.yaml         toutes les valeurs configurables
data/{raw,processed,results} Parquet (ignorés par git)
src/rte_forecast/
  data/            loaders, cleaning, imputation causale, température, pipeline, synthetic (tests)
  calendar/        holidays, bridge_days, special_periods
  features/        build_features (SANS fuite)
  models/          baseline (J-7, RTE J-1), sarimax, xgboost_model, random_forest
                   tft (Temporal Fusion Transformer compact, torch : extra `.[tft]`, non importé par défaut)
  business_rules/  analogue, period_transfer, year_end, year_start, summer, isolated_holidays, weather
  evaluation/      splits, metrics, backtest, analysis
  forecast.py      routeur RF/XGB  <->  règles métier + explicabilité
app/dashboard.py   dashboard Streamlit
scripts/           01_prepare_data, 02_analyze_hypotheses, 03_run_benchmark, 04_bootstrap_compare, 05_train_tft, 06_merge_tft
tests/             anti-fuite, qualité des données, calendrier, règles métier, métriques
```

## Convention temporelle (anti-fuite)

La prévision du jour D est émise à **D 00:00** : la dernière charge connue est celle de D-1 23:00.
Toute feature dérivée de la charge a donc un **retard ≥ 24 h** (`build_features` refuse moins). Les
moteurs de règles ne lisent que des données antérieures au début de la période prévue (mode
`planning`) ou au jour prévu (mode `adaptive`). Les splits sont strictement chronologiques,
jamais de shuffle. Voir `tests/test_features_leakage.py` et `tests/test_business_rules.py`.

## Scénarios météo (à lire avant d'interpréter un résultat)

| mode | température « prévue » | statut |
|---|---|---|
| `normal` | normale Enedis | opérationnel de référence (aucune vraie prévision disponible) |
| `noisy` | réalisée + bruit AR(1) | **SYNTHÉTIQUE** : scénario réaliste, à remplacer par des prévisions archivées |
| `perfect` | réalisée | **ORACLE** : plafond théorique, jamais un résultat opérationnel |

Sources, limites et résultats : voir `CHECKPOINT.md` et `RESULTS.md`.
