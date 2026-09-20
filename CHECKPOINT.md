# CHECKPOINT — rte-load-forecasting-demo

> Journal de bord pour reprendre le travail. Dernière mise à jour : 2026-09-20 (benchmark complet relancé en convention D-1 10:00, modèles sauvegardés).
> Les résultats chiffrés et leur interprétation sont dans **`RESULTS.md`** ; ici : ce qui a été fait,
> comment, ce qui a coincé, et ce que vous devez remplacer/valider.

## 0. À FAIRE PAR VOUS (données de remplacement à signaler)

| # | Élément | Statut | Où c'est marqué | À remplacer par |
|---|---|---|---|---|
| 1 | Température réalisée **2020-01-01 → 2021-09-18** (15 048 h) | **PROXY** ERA5/Open-Meteo (12 villes pondérées, EWMA 24 h, recalé linéairement sur Enedis ; RMSE 1,00 °C, corr. 0,987 sur 3,3 ans de recouvrement) | colonne `temperature_source = openmeteo_calibrated` dans `data/processed/hourly.parquet` | Données Enedis d'avant sept. 2021 si vous en obtenez (l'API n'expose qu'une fenêtre glissante de 5 ans) |
| 2 | Normale de température 2020-01 → 2021-09 | Projetée depuis la normale Enedis (identique d'une année à l'autre, vérifié) | idem | — |
| 3 | Prévision de température « réaliste » (scénario `noisy`) | **SYNTHÉTIQUE** : réalisée + erreur AR(1) (σ=1,2 °C, φ=0,6, RMSE 1,24 °C) | `weather_mode = noisy` | Prévisions météo archivées (Open-Meteo historical-forecast, Météo-France AROME/ARPEGE, ECMWF) |
| 4 | Prévision de température opérationnelle (`normal`) | = normale Enedis, comme demandé (aucune source de prévision exploitable rapidement) | `weather_mode = normal` | idem |
| 5 | Jeux de données pour les tests | Série **synthétique** déterministe (`src/rte_forecast/data/synthetic.py`) | tests uniquement, jamais dans les résultats | — |

Aucune donnée de **charge** n'est synthétique : eCO2mix 2020-2024 est complet (0 valeur manquante).

## 1. Ce qui a été fait (dans l'ordre de la spec §18)

1. **Analyse des données** — 5 zips eCO2mix (TSV latin-1 déguisé en `.xls`, pas 15 min mais consommation seulement aux pas :00/:30, ligne de disclaimer finale, colonne vide finale) → 43 848 h, 0 doublon, 0 NaN. `data/raw/*.parquet`, `data/processed/hourly.parquet`, `data_quality.json`.
2. **Calendrier** (`src/rte_forecast/calendar/`) — fériés (bibliothèque `holidays`, 11/an), ponts (définition stricte de la spec), S51–S53 = YEAR_END, S1–S2 = YEAR_START, fenêtre estivale, précédence YEAR_END > YEAR_START > SUMMER > HOLIDAY > BRIDGE > NORMAL.
3. **Vérification empirique des hypothèses** — `scripts/02_analyze_hypotheses.py` (voir § 3 ci-dessous : **deux hypothèses de la spec sont contredites ou nuancées**).
4. **Features sans fuite** — retard ≥ 24 h imposé par le code (`min_lag_hours<24` lève une erreur), test d'invariance « perturber la charge dès D 00:00 ne change aucune feature du jour D ».
5. **Baselines/modèles** — J-7, RTE J-1 (référence externe), SARIMAX (exogènes T°, jour de semaine, férié, pont), XGBoost, Random Forest (chacun « tous jours »).
6. **Moteurs de règles** — fin d'année, début d'année, été, fériés/ponts isolés, routeur `forecast.py`, explicabilité (`format_explanation`).
7. **Backtest walk-forward** — 12 folds trimestriels (2022-T1 → 2024-T4), split train ≤ 2022 / validation 2023 / test 2024, grille d'hyperparamètres sur la validation, SHAP.
8. **Dashboard Streamlit** (`app/dashboard.py`) — choix du modèle, jeux de données, heatmaps annuelles (x = jour, y = mois), distributions d'erreurs, prévisions vs réel, CV, hyperparamètres, importance/SHAP, règles métier + explications.
9. **Tests (102) + CI GitHub Actions (ruff → pytest)** ; test de mutation manuel : une fuite injectée est bien détectée.

## 2. Décisions de conception à connaître

* **Instant d'émission (convention unique)** : le forecast du jour D est émis à **D-1 10:00** (charge connue jusqu'à D-1 09:00, `features.issue_hour: 9`), comme une vraie J-1. Lag 48 h pour les heures cibles > 9, moyennes glissantes ancrées à D-1 09:00, SARIMAX et règles adaptatives décalés (D-8..D-2). Une première version émettait à D 00:00 (optimiste, ≈ −24 % de précision) : abandonnée. L'heure de 10:00 reste une hypothèse à confronter au calendrier de publication RTE.
* **Deux versions de features** : « brutes » (modèles tous jours) et « propres » (lags dont la *source* est un jour atypique neutralisés → NaN, repli hebdomadaire jusqu'à 6 semaines) pour le modèle « jour normal ». Les NaN sont gérés nativement (XGBoost, sklearn ≥ 1.4).
* **Température** : trois scénarios (`normal` opérationnel, `noisy` synthétique, `perfect` oracle). La température **réalisée n'est jamais une feature** en `normal`/`noisy` (test dédié). Le mode `perfect` est un plafond théorique, pas un résultat.
* **« RF seul »/« XGBoost seul »** = entraînés sur *tous* les jours avec les drapeaux calendaires (comparaison équitable) ; **hybride** = entraîné sur jours NORMAUX uniquement + règles pour le reste.
* **Règles — deux modes de niveau** :
  * `planning` (fidèle à la spec) : niveau ancré sur S48–S50, période prévue 1 à 3 semaines à l'avance ;
  * `adaptive` : même forme, mêmes coefficients, mais niveau ré-ancré chaque jour sur les 7 derniers jours observés. Ajouté parce que la spec `planning` compare des prévisions à horizon long à un RF *day-ahead* avec lags → comparaison à information inégale (voir RESULTS.md).
* **Normalisation météo des références** (`business_rules/weather.py`) : non prévue par la spec, ajoutée après constat que transférer des ratios bruts importe l'anomalie de température de l'année analogue (décembre 2022 : S51 à 0,81 de la baseline contre 0,96 en 2023). Sensibilité thermique estimée *causalement* sur l'historique (~2,0 GW/°C).
* **Année analogue** : score = 1 − (0,5·Δjour_semaine/3 + 0,2·Δsemaine_ISO/2 + 0,3·Δstructure W/H/B/O), égalités → année la plus récente ; années COVID écartées tant qu'il reste des candidats.
* **Alignement des semaines** : lundi le plus proche de la même position relative à l'ancre (Noël/Nouvel An) ; 25/12 et 1/01 appariés par identité de date.
* **Fériés isolés** : rien d'arbitraire — coefficient = médiane des ratios historiques (charge du jour / même jour de semaine ordinaire à ±7 j), sélection par paliers (même férié+même classe de jour → même classe → même férié). Une occurrence n'est utilisable que si **toutes** ses données (jour + voisins) précèdent le jour prévu (test dédié).
* **Fenêtre estivale = ±2 semaines** (et non ±4) : les données ne montrent aucune baisse avant −1 semaine.

## 3. Hypothèses de la spec confrontées aux données (signalées, non masquées)

| Hypothèse | Verdict | Détail |
|---|---|---|
| Le creux estival est la semaine dont le week-end est le plus proche du 15 août | **Partiellement vérifiée** | Exacte 2023, 2024 (et 2020 sur juil–sept.) ; décalée de −1 sem. en 2021, +1 sem. en 2022 (le candidat est 2,2 % / 4,9 % au-dessus du minimum) ; **fausse en 2020 sur juin–sept.** (minimum au 1ᵉʳ juin, post-confinement) |
| Le ramp-down estival est progressif (plusieurs semaines) | **Contredite** | Creux net et bref : −6 % sur ~3 semaines autour de la semaine du creux ; aucune baisse mesurable avant −1 semaine → fenêtre réduite à ±2 |
| Noël et Nouvel An ont des jours de semaine distincts | **Trivialement fausse** | Ils tombent *toujours* le même jour de semaine (7 jours d'écart) |
| S51/S52 = fin d'année | **Insuffisant** | Certaines années ont une S53 (2020, 2026) et Noël/1ᵉʳ janvier peuvent tomber en S51/S52/S53/S1 → périodes définies par semaines ISO {51,52,53} et {1,2}, avec cas limites documentés |
| Transférer des *ratios* suffit | **Nuancé** | Il faut aussi normaliser la météo des références (cf. § 2) |

## 4. Difficultés rencontrées

1. **Enedis n'a que 5 ans glissants** (depuis 2021-09-18) alors que eCO2mix démarre en 2020 → proxy Open-Meteo calibré (cf. § 0).
2. **Rate-limit Open-Meteo** : timeout en pipeline après mes tests → retries avec backoff + cache Parquet + repli synthétique signalé.
3. **Données propres** : 0 valeur manquante → le code d'imputation n'est jamais exercé en production ; validé par injection de trous dans les vraies données (MAPE 1,2 % / 3,5 % / 4,4 % pour 3 / 24 / 72 h) et par tests.
4. **Erreurs énormes au premier essai des règles** (6–15 GW, biais systématique) : ce n'était pas un bug mais l'anomalie météo de l'année analogue → normalisation météo.
5. **Année analogue médiocre** : avec 5 ans d'historique, le meilleur analogue a souvent un score de 0,25–0,5 (jour de semaine de Noël différent) ; 2021-2023 sont marquées par la crise énergétique (sobriété) et le COVID.
6. **pandas 3** (dtype `str` strict) : colonnes texte `object`, entiers/`None` mélangés illisibles en Parquet, `None` YAML lu comme chaîne (→ `null`).
7. **Mémoire limitée (7 Go)** : le dernier run a été fait sur 1 seul processus (`--jobs 1`, `nice 19`, 1 thread BLAS) pour ne pas encombrer la machine (72 min) ; les modèles sont sauvegardés dans le processus qui les entraîne. Résultats identiques (0,0 MW) à ceux d'un run à 4 processus.
8. **Meilleurs hyperparamètres en bordure de grille** (XGBoost : profondeur 9 = max, 500 arbres = max ; RF : `min_samples_leaf=2` = min) → la grille devrait être étendue.
9. **Playwright MCP indisponible** : le dashboard n'a pas pu être validé visuellement dans un navigateur ; il est vérifié via `streamlit.testing.AppTest` (aucune exception) — **à regarder à l'œil**.

## 5. Résultat en une phrase

Sur le test 2024 (émission D-1 10:00, météo opérationnelle = normale), **RF/XGBoost « tous jours » (MAE 1,70 GW) battent SARIMAX (2,91 GW) et J-7 (3,23 GW)** mais restent derrière **RTE J-1 (1,43 GW)**, et **le hybride ML + règles n'est pas meilleur que le ML seul** (1,94–1,98 GW) : leur forme horaire est meilleure mais leur niveau est biaisé par l'anomalie de température (§ 3 bis de `RESULTS.md`) ; seuls les fériés isolés bénéficient des règles (et toutes les règles bénéficient d'une météo connue). Détails, IC et limites : `RESULTS.md`.

## 6. Prochaines étapes

Voir `RESULTS.md § 6`. Priorités : (1) vraies prévisions météo archivées (ou au moins une persistance de l'anomalie de température) pour le ML et les règles, (2) confronter l'heure d'émission (10:00) au calendrier réel de publication RTE, (3) historique eCO2mix 2012-2019, (4) go/no-go sur les règles (garder les fériés isolés).

## 7. Reproduire

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev,dashboard]"
.venv/bin/python scripts/01_prepare_data.py    # ⚠ retélécharge Enedis (fenêtre glissante de 5 ans) : NE PAS relancer si on veut retrouver les résultats publiés
.venv/bin/python scripts/02_analyze_hypotheses.py
OMP_NUM_THREADS=1 nice -n 19 .venv/bin/python scripts/03_run_benchmark.py --jobs 1 --save-models   # ~72 min sur 1 cœur ; --jobs 4 : ~35 min
.venv/bin/python scripts/04_bootstrap_compare.py --a random_forest_rules_adaptive --b random_forest
.venv/bin/streamlit run app/dashboard.py
.venv/bin/pytest && .venv/bin/ruff check .
```
