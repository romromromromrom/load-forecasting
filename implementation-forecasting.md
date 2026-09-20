# implementation-forecasting — journal de la conversation

> Projet : `rte-load-forecasting-demo` (prévision de la consommation française, RTE eCO2mix + Enedis).
> Session du 2026-09-19 au 2026-09-20. Les messages de l'utilisateur sont reproduits tels quels ;
> les réponses de l'assistant sont reproduites fidèlement, légèrement condensées pour les messages
> d'attente et de suivi de tâches en arrière-plan (moniteur, relances). Les sorties d'outils brutes
> ne sont pas incluses, seulement leurs résultats utiles.

## Sommaire

1. [Récap du repo](#1-récap-du-repo)
2. [Conformité au prompt initial](#2-conformité-au-prompt-initial)
3. [Audit : modification de `build_features.py` et instant d'émission](#3-audit--modification-de-build_featurespy-et-instant-démission)
4. [Les PCA ?](#4-les-pca-)
5. [Gestion du changement d'heure](#5-gestion-du-changement-dheure)
6. [Le lag 168 h autour du changement d'heure](#6-le-lag-168-h-autour-du-changement-dheure)
7. [Test de verrouillage du changement d'heure](#7-test-de-verrouillage-du-changement-dheure)
8. [Coût d'un Transformer temporel (temps / RAM)](#8-coût-dun-transformer-temporel-temps--ram)
9. [Et sur Colab / Kaggle ?](#9-et-sur-colab--kaggle-)
10. [Durcissement de l'émission (implémentation + benchmark)](#10-durcissement-de-lémission-implémentation--benchmark)
11. [Script TFT pour Google Colab](#11-script-tft-pour-google-colab)
12. [Commit et push GitHub (interrompu)](#12-commit-et-push-github-interrompu)
13. [Ce fichier](#13-ce-fichier)
14. [Annexe : état final du projet](#annexe--état-final-du-projet)

---

## 1. Récap du repo

**Utilisateur :**
> lis tout le repo, les résultats et les markdowns et fait moi un récap de ce qui a été testé implémenté et quels sont les key takeways, aussi essaie de me résumer dans quel ordre les choses ont été implémentées

**Assistant :**

Récap après lecture du code, des configs, de `CHECKPOINT.md`, `RESULTS.md`, `README.md`, `prompt-init.md` et des parquets de `data/results/`. `pytest` (68 tests) et `ruff` relancés : tout passe. Tous les fichiers du projet datent du 2026-09-19 entre 14:59 et 16:52.

**Périmètre.** Démonstrateur de prévision de la consommation française (eCO2mix RTE 2020-2024 et température Enedis), pipeline simple, testé et reproductible, sans Airflow, Docker ni MLflow. Environ 3 700 lignes de Python.

| Bloc | Contenu |
|---|---|
| **Données** | 5 zips eCO2mix (TSV latin-1 déguisés en `.xls`), 43 848 h sans doublon ni manquant. Température Enedis via API. Nettoyage, imputation causale, cache Parquet. |
| **Calendrier** | Fériés, ponts (définition stricte de la spec), périodes YEAR_END (S51-S53), YEAR_START (S1-S2), SUMMER (±2 semaines autour du creux), HOLIDAY. |
| **Features** | Lags ≥ 24 h imposés par le code, rolling means, température au carré, interactions calendaires. Deux versions : « brutes » et « propres » (lags issus de jours atypiques neutralisés). |
| **Modèles** | J-7, RTE J-1 (référence externe), SARIMAX, XGBoost, Random Forest. |
| **Règles métier** | `year_end`, `year_start`, `summer`, `isolated_holidays`, `analogue`, `period_transfer` (niveau → semaine → jour → profil), `weather` (normalisation météo, ajoutée par l'assistant). Routeur `forecast.py` et explicabilité. |
| **Évaluation** | Split train ≤ 2022 / validation 2023 / test 2024, 12 folds walk-forward trimestriels, grille d'hyperparamètres, SHAP, MAE/RMSE/biais par période, bootstrap apparié. |
| **Livrables** | Dashboard Streamlit, 68 tests dont anti-fuite, CI GitHub Actions (ruff puis pytest), README, CHECKPOINT, RESULTS. |

**Ordre d'implémentation** (reconstitué à partir des dates de modification, le dossier n'étant pas commité ; `CHECKPOINT.md §1` liste plutôt l'ordre de la spec §18) :

1. 14:59-15:04 : dépôt des zips eCO2mix, du logo, puis `prompt-init.md` (15:40).
2. 15:52-15:56 : squelette (`pyproject`, config), puis pipeline de données (loaders, cleaning, Enedis, proxy Open-Meteo, `hourly.parquet`).
3. 15:58-16:02 : calendrier.
4. 16:03 : modèles (baseline, SARIMAX, XGBoost, RF).
5. 16:06-16:17 : règles métier (`analogue`, `isolated_holidays`, puis `period_transfer`, `year_end`, `year_start`, `summer`, `weather`, routeur). La normalisation météo est arrivée après des erreurs de 6 à 15 GW sur les premiers essais.
6. 16:12-16:21 : `metrics`, `splits`, `backtest`, script `03_run_benchmark.py`.
7. 16:19-16:22 : config YAML, données synthétiques, tests, imputation, `02_analyze_hypotheses.py`.
8. ~16:22-16:48 : benchmark (hyperparamètres, puis 3 scénarios météo de 24 tâches, puis SHAP).
9. 16:26-16:27 : CI et README ; dernière modification de `build_features.py` à 16:27, pendant le benchmark.
10. 16:50-16:52 : dashboard, parquets d'analyse, CHECKPOINT et RESULTS.

**Ce qui a été testé.** Imputation (MAPE 1,2 % / 3,5 % / 4,4 % pour des trous de 3 / 24 / 72 h) ; fuite de données (test d'invariance : perturber la charge dès D 00:00 ne change aucune feature du jour D, plus un test de mutation manuel) ; hypothèses de la spec ; trois scénarios météo (`normal` = T° prévue = normale Enedis, `noisy` = réalisée + bruit AR(1) synthétique, `perfect` = oracle) ; 9 variantes de modèles dont règles `planning` et `adaptive`.

**Key takeaways** (test 2024, météo `normal`, MAE en MW) :

| Modèle | MAE |
|---|--:|
| J-7 | 3 229 |
| SARIMAX | 2 018 |
| XGBoost | 1 377 |
| Random Forest | **1 343** |
| RF + règles | 1 665-1 711 |
| *RTE J-1* | *1 428* |

1. XGBoost et RF battent SARIMAX de 32 % et J-7 de 57 %. XGBoost s'entraîne 7 fois plus vite que SARIMAX (9 s contre 67 s par fold). Aucune raison technique d'ajouter du deep learning.
2. Le lag 24 h domine le signal (≈ 52 % d'importance, SHAP 3,7 fois celui de la 2ᵉ variable). La température prévue pèse très peu en `normal`.
3. L'hypothèse centrale « ML + règles > ML seul » est infirmée : l'hybride est significativement moins bon sur l'été et S51-S2 en `normal`, indiscernable en `noisy`.
4. Seul succès des règles : les fériés isolés (−545 MW, IC 95 % [−1 073 ; −59] en `noisy`), avec un coefficient mesuré (médiane des ratios historiques).
5. Le facteur limitant est la météo, pas le calendrier : avec une météo parfaite, le moteur fin/début d'année tombe à 1,4-2,6 GW de MAE contre 4-11 GW pour J-7.
6. Exclure les jours spéciaux de l'apprentissage n'aide pas (1 330 contre 1 273 MW sur les jours normaux).
7. Hypothèses de la spec contredites ou nuancées : le creux estival tombe la semaine du 15 août seulement 2 années sur 5 exactement ; pas de rampe estivale progressive (creux de ~3 semaines, −6 %) ; Noël et Nouvel An tombent toujours le même jour de semaine ; S51-S53 plutôt que S51/S52 (certaines années ont une S53).

**Limites.** Instant d'émission optimiste (D 00:00, charge de la veille complète) ; pas de vraie prévision météo ; température 2020-01 → 2021-09 = proxy ERA5 recalé (RMSE 1 °C) ; historique court (5 ans, COVID en 2020, crise énergétique 2022-23) : seulement 5 ponts, 24 fériés, 44 jours de S51-S53 et 44 de S1-S2 sur la période d'évaluation ; hyperparamètres en bordure de grille ; dashboard non contrôlé visuellement (Playwright indisponible).

**Deux points non confirmés à ce moment-là :** `build_features.py` modifié à 16:27 pendant le benchmark (fini à 16:48) ; et le dossier `load-forecasting-ml` est ignoré par le `.gitignore` du repo parent (règle sur `energy-job-radar/applications/`), donc le projet n'est versionné nulle part.

**Prochaines étapes** (`RESULTS.md §6`) : vraies prévisions météo archivées ; émission plus réaliste (D-1 10:00) ; historique eCO2mix 2012-2019 ; go/no-go sur les règles (garder les fériés isolés) ; grille élargie, quantiles, LightGBM ; seulement ensuite Airflow, Docker, MLflow.

---

## 2. Conformité au prompt initial

**Utilisateur :**
> est-ce que ça correspond bien aux instructions données dans le prompt-ini.md ?

**Assistant :**

Globalement oui : toutes les sections du prompt ont un équivalent dans le repo, avec quelques écarts documentés. Comparaison faite entre le prompt et le code (features, noms des tests, dashboard) ; pas de relecture ligne par ligne de chaque module.

**Conforme**

| Demande du prompt | État |
|---|---|
| Données : zips eCO2mix, API Enedis, prévision = normale si rien d'exploitable, Parquet, sources documentées | ✅ |
| Features : lags 24/48 h, même jour S-1 et S-2, rolling means, T° normale et réalisée, carrés, interactions | ✅ `load_lag_24h/48h/168h/336h`, `rolling_mean_24h/7d`, `temp_x_weekend`, `temp_x_hour`… (voir écart 1) |
| Baselines : J-7, SARIMAX avec T° exogène, XGBoost, XGBoost + règles | ✅ plus RF, RF + règles et RTE J-1 |
| Validation : pas de split aléatoire, train/val/test chronologiques, CV expanding, MAE/RMSE/biais/temps | ✅ deux tests l'imposent |
| Visualisations : erreurs, MAE par heure, folds CV, temps, hyperparamètres, importance, SHAP summary et dependence température | ✅ dans le dashboard Streamlit (graphiques interactifs, lisibilité non jugée visuellement) |
| Architecture, CI (ruff puis pytest), RESULTS.md, pas d'Airflow/Docker/MLflow | ✅ |
| Tests d'hygiène : doublons, NaN cible, incohérence des lags, train après test | ✅ un test par cas |
| Les 9 tests métier obligatoires | ✅ retrouvés par leur nom |
| Moteur de règles complet, routeur, explicabilité | ✅ ; fériés isolés mesurés, pas arbitraires |
| « Si une règle est contredite, ne pas la corriger silencieusement » | ✅ contradictions signalées dans `CHECKPOINT.md §3` |
| Tableau de benchmark par période | ✅ `RESULTS.md §3` |

**Écarts**

1. **T° réalisée et normale comme features** : elles ne sont dans le modèle qu'en mode `perfect` (oracle) ; en mode opérationnel seule la T° « prévue » entre, un test interdit la réalisée. Cohérent avec l'anti-fuite mais pas une lecture littérale du prompt.
2. **Périodes redéfinies** : S51-S53 au lieu de S51/S52 ; fenêtre estivale ±2 semaines. Argumenté par les données, fait explicitement.
3. **Ajouts non demandés** : normalisation météo des références, mode `adaptive`, 3 scénarios météo, référence RTE J-1, bootstrap. Plus de complexité que « ne complexifie pas artificiellement ».
4. **« RF seul » ambigu** : RF « tous jours » et « jours normaux » sont en concurrence ; le résultat contredit l'argument du prompt.
5. **Ordre §18 non vérifiable** : `02_analyze_hypotheses.py` a une date de modification (16:22) postérieure aux fichiers de règles (16:06-16:17) ; les dates ne prouvent rien mais l'ordre affirmé n'est pas confirmé.
6. **Instant d'émission** : D 00:00 avec la charge de la veille complète, fidèle au « lag 24 h » demandé mais optimiste par rapport à une vraie J-1.

**À faire de votre côté** : l'audit de fuite prévu à la fin du prompt. Deux points prioritaires : la modification de `build_features.py` à 16:27 pendant le benchmark, et l'émission à D 00:00.

---

## 3. Audit : modification de `build_features.py` et instant d'émission

**Utilisateur :**
> ok fais le :
>
> La fin du prompt prévoyait un audit de fuite de 30 min à votre retour. Deux points à regarder en priorité :
> - build_features.py a été modifié à 16:27, pendant le benchmark qui finit à 16:48. Relancer le benchmark ou comparer les prédictions permettrait de le confirmer.
> - L'émission à D 00:00 (écart 6).

**Assistant :**

Trois vérifications, dans un script à part (scratchpad, repo intact).

**1. `build_features.py` modifié à 16:27 : sans effet sur les résultats.** XGBoost refait tourner avec le code actuel sur les 12 folds, dans les trois scénarios météo : écart avec `predictions.parquet` de **0,000000 MW** pour XGBoost « tous jours » (`normal`, `noisy`, `perfect`) et « jours normaux » (`normal`). Les résultats stockés correspondent donc exactement au code actuel (RF, SARIMAX et règles non rejoués, mais mêmes matrices de features).

**2. Fuite de données : aucune trouvée.** Pour 30 jours tirés au hasard entre 2022 et 2024, la charge est mise à NaN à partir de D 00:00 et les features reconstruites : les 24 h de D sont identiques sur toutes les colonnes, en version brute et propre.

**3. Instant d'émission D 00:00 : le résultat le plus important.** Le lag 24 h n'est utilisable pour toutes les heures de D qu'en émettant à D 00:00, ce qui est irréaliste. Simulation d'une émission à **D-1 10:00** (charge connue jusqu'à D-1 09:00) : lag 24 h conservé pour les heures cibles 0 à 9, lag 48 h pour les heures 10 à 23, rolling means ancrées à D-1 09:00. Hyperparamètres non ré-optimisés. MAE test 2024 :

| Modèle (météo) | Actuel | Émission D-1 10:00 | Écart |
|---|--:|--:|--:|
| XGBoost (`normal`) | 1 377 | **1 704** | +24 % |
| Random Forest (`normal`) | 1 343 | **1 703** | +27 % |
| XGBoost (`noisy`) | 1 295 | 1 544 | +19 % |
| RF (`noisy`) | 1 287 | 1 569 | +22 % |
| XGBoost (`perfect`) | 1 130 | 1 321 | +17 % |
| *RTE J-1* | *1 428* | *1 428* | |
| *J-7* | *3 229* | *3 229* | |

- La dégradation vient des heures 10 à 23 (MAE ≈ 1 300-1 400 → ≈ 1 800-2 000 MW).
- L'affirmation « RF au niveau de RTE J-1 » tombe : en `normal`, XGBoost et RF font ~275 MW de plus que RTE. Les modèles ML restent à ~47 % sous J-7.
- Effet sur l'hybride (à confirmer) : sur les périodes spéciales les règles ne dépendent pas de l'émission, donc les fériés (1 301 contre 1 438 / 1 943) et S51-S53 (3 375 contre 3 507 / 3 733) passent légèrement devant le ML, tandis que S1-S2 et l'été restent nettement en défaveur des règles. La conclusion « les règles n'apportent rien » est sensible à l'instant d'émission.

**Limites de l'audit :** l'heure de 10:00 est une hypothèse ; hybrides, mode `adaptive` et SARIMAX non rejoués avec l'émission durcie à ce stade. Aucun fichier du repo modifié ; scripts dans le scratchpad (`audit_ab.py`, `audit_c.py`).

---

## 4. Les PCA ?

**Utilisateur :**
> est-ce qu'on a fait les PCA ?

**Assistant :**

Non, aucune PCA (ni ACP, SVD ou autre réduction de dimension) : une recherche dans tout le code, les configs et les markdowns ne renvoie rien.

- **Le prompt ne la demandait pas.**
- **Elle n'apporterait probablement pas grand-chose** : une vingtaine de features (12 de calendrier, 5 de météo, 6 de charge), XGBoost et RF gèrent bien des features corrélées, et une PCA nuirait à l'explicabilité (SHAP, règles, importance par variable).
- **Pistes où elle pourrait servir** : (1) météo multi-villes (12 villes pondérées dans le proxy Open-Meteo → 2 ou 3 facteurs thermiques régionaux, utile seulement avec de vraies prévisions par ville) ; (2) analyse des profils journaliers normalisés (24 h) pour regrouper les types de jours et affiner le choix de l'année analogue ; (3) diagnostic de redondance des features.

---

## 5. Gestion du changement d'heure

**Utilisateur :**
> comment est géré le changement d'heure ?

**Assistant :**

Le changement d'heure n'est pas géré explicitement : le projet le contourne en travaillant en heure locale « murale » à 24 pas par jour (hypothèse en docstring, `cleaning.py:14`). Aucun test ne le couvrait.

**Ce que fait le code**
- Index sans fuseau (`tz = None`), 24 lignes par jour, y compris les 10 jours de changement d'heure de 2020-2024.
- Fichiers eCO2mix : les 96 pas sont lus tels quels (vérifié : 96 lignes par jour pour les changements d'heure de 2022 et 2024).
- Température : Enedis et Open-Meteo arrivent en UTC, convertis en `Europe/Paris` puis rendus naïfs ; heures dupliquées d'automne moyennées, heure manquante du printemps interpolée.
- Lags et fenêtres en nombre de lignes : `shift(24)` = même heure d'horloge la veille, `shift(168)` = même jour la semaine précédente.

**Ce que ça donne dans les données**
- **Printemps** : l'heure inexistante est comblée par une copie (02:00 = 03:00 : 46 130 MW le 2022-03-27, 44 920 le 2023-03-26, 47 004 le 2024-03-31) ; le fichier brut répète aussi ces valeurs, c'est RTE qui remplit. Seules valeurs consécutives identiques de la série (avec 5 autres cas ; 2020 non cherché).
- **Automne** : l'heure répétée n'apparaît qu'une fois, une heure d'information est perdue ; chute brutale à 03:00 (36 150 → 31 690 le 2022-10-30). Inférence, pas confirmée par la documentation RTE.

**Conséquences** : faibles (~10 jours sur 1 827, 0,5 %), pas de fuite, non détecté par `detect_invalid`, pas de test, lags décalés d'une heure physique ces jours-là ; erreur des modèles ces jours-là non mesurée.

---

## 6. Le lag 168 h autour du changement d'heure

**Utilisateur :**
> typiquement quand je fais du j -168h pour trouver le même jour la semaine précédente ça ne marche plus quand on est autour du changement d'heure

**Assistant :**

Ici, le lag 168 h marche autour du changement d'heure, parce que l'index est en heure locale naïve. Votre remarque est juste dès que l'index est en UTC ou tz-aware. Vérifié sur `hourly.parquet` :

- Index régulier, pas de 1 h, sans fuseau, 24 lignes par jour.
- `shift(168)` donne toujours exactement 7 jours d'écart entre horodatage cible et source.
- Sur toutes les lignes, la source a la même heure d'horloge et le même jour de semaine que la cible, y compris autour des 10 changements d'heure.

**Avec un vrai temps écoulé** (index localisé en `Europe/Paris`, 168 h réelles) : 1 675 lignes sur 43 838 (3,8 %) tombent sur une autre heure d'horloge, soit les 7 jours qui suivent chacun des 10 changements d'heure. Le lundi 08:00 après le passage à l'heure d'été irait chercher le lundi précédent à 07:00, avec un profil de charge très différent.

**Recommandation** : rester en heure murale (la charge suit les habitudes humaines), au coût de la copie au printemps et de l'heure perdue à l'automne ; sur un index tz-aware, utiliser `ts - pd.DateOffset(days=7)` plutôt que `Timedelta(hours=168)` ou `shift(168)`.

---

## 7. Test de verrouillage du changement d'heure

**Utilisateur :**
> oui

**Assistant :**

Ajout de `tests/test_dst.py` : 14 tests passent, `ruff` propre, suite complète à 82 tests (68 avant). Les tests utilisent la série synthétique (CI sans les données réelles) ; la charge de test encode son propre horodatage pour vérifier d'où vient chaque lag.

| Test | Ce qu'il verrouille |
|---|---|
| `test_index_is_naive_regular_and_has_24_rows_on_dst_days` | Index naïf, pas régulier de 1 h, 24 lignes par jour, y compris les 10 jours de changement d'heure de 2020-2024 (derniers dimanches de mars et d'octobre) |
| `test_weekly_lags_keep_clock_hour_and_weekday_across_dst` | Pour les 4 lags de charge, la source est à `t − N h` ; pour 168 et 336 h, même heure d'horloge et même jour de semaine |
| `test_lag_168h_is_the_same_clock_hour_last_week_around_dst` (×10) | Pour chaque changement d'heure, sur la fenêtre veille → +8 jours, exactement 7 jours d'écart |
| `test_persistence_j7_uses_the_same_clock_hour_across_dst` | Même propriété pour J-7 |
| `test_elapsed_time_arithmetic_on_tz_aware_index_would_misalign` | Documente le piège : le lundi 2024-04-01 à 08:00, `Timedelta(hours=168)` renvoie 07:00 alors que `DateOffset(days=7)` renvoie 08:00 |

Ces tests protègent l'existant : si l'index passe en UTC ou tz-aware, ou si le calcul des lags change, l'un d'eux échoue. Ils ne changent rien à la copie de l'heure au printemps ni à l'heure perdue à l'automne.

---

## 8. Coût d'un Transformer temporel (temps / RAM)

**Utilisateur :**
> combien de temps / de ram prendre l'implémentation des time series transformers dans ce cas ?

**Assistant :**

Estimations (PyTorch non installé, rien de mesuré). Machine : 8 cœurs, 7,4 Go de RAM (~3,6 Go disponibles), pas de GPU.

**Modèle visé** : petit Transformer encodeur type PatchTST, contexte de 168 à 336 h en patchs de 8 h (21 à 42 jetons), d_model 64, 3 couches, 4 têtes, ~0,3 M de paramètres, sortie 24 h avec calendrier et température prévue en covariables futures. Sans patchs (168 jetons), environ 8 fois plus lent.

**Temps** (8 000 à 41 000 fenêtres d'entraînement selon le fold) :

| Poste | Estimation |
|---|---|
| 1 epoch, CPU, avec patchs | 20 à 60 s (5 à 10 min sans patchs) |
| 1 fit (20-30 epochs, arrêt précoce) | 10 à 20 min |
| 12 folds, un scénario météo | 2 à 4 h |
| 12 folds × 3 scénarios | 6 à 10 h |
| Recherche d'hyperparamètres (6 configs) | ~1 à 2 h |
| Plusieurs graines (×3) | ×3 |

Pour comparaison, le benchmark actuel prend ~25 min (XGBoost 9 s par fold, RF 35 s, SARIMAX 67 s). Réductions possibles : un seul scénario, ré-entraînement annuel (3 fits, ~40 min), fine-tuning du fold précédent (~30 min). Sur GPU grand public : 10 à 30 fois moins.

**RAM** : données minuscules (quelques Mo ; ~0,5 Go si on matérialise toutes les fenêtres, donc à construire à la volée) ; 1 à 2 Go par processus d'entraînement ; les 4 processus du benchmark actuel dépasseraient les 3,6 Go disponibles → 1 à 2 processus, donc plus long.

**Temps de développement** : une demi-journée à une journée (module de modèle, fenêtres sans fuite avec tests, intégration au backtest et au dashboard, installation de `torch`).

**Avis** : gain attendu faible (le lag 24 h porte l'essentiel du signal, 5 ans de données, XGBoost bat déjà J-7 de 57 %) ; les Transformers battent rarement les arbres boostés en prévision de charge à un jour dans ces conditions, ce qui rejoint la position du prompt : pas de deep learning tant que le gain hors échantillon ne justifie pas la complexité. Ordre raisonnable : émission durcie et vraies prévisions météo d'abord, puis petit Transformer sur un seul scénario avec ré-entraînement annuel (~1 h de calcul).

---

## 9. Et sur Colab / Kaggle ?

**Utilisateur :**
> et si je le fais sur google collabs ou sur kaggle puis que je télécharge le modèle entraîné ensuite ?

**Assistant :**

Bonne idée : ça supprime le principal obstacle, le temps de calcul (estimations, pas mesurées).

| | Local (CPU) | Colab ou Kaggle (GPU T4/P100) |
|---|---|---|
| 1 fit (20-30 epochs) | 10 à 20 min | 1 à 3 min |
| 12 folds × 3 scénarios | 6 à 10 h | ~1 à 2 h |
| Avec 3 graines | ×3 | ~3 à 5 h |
| RAM | ~1-2 Go par processus, ~3,6 Go dispo | ~12 Go (Colab gratuit) ou ~29 Go (Kaggle), plus 16 Go de VRAM |

Limites de plateforme (connaissance à vérifier) : Colab gratuit, sessions d'environ 12 h, déconnexion si inactif, GPU non garanti ; Kaggle, quota GPU hebdomadaire d'environ 30 h et sessions d'environ 9 h. Dans les deux cas, écrire les résultats fold par fold.

**Télécharger les prédictions, pas seulement les poids** : les poids (1 à 4 Mo par modèle) ne servent à rien seuls pour le benchmark (12 checkpoints et PyTorch en local). Un `predictions_tft.parquet` au même schéma que `predictions.parquet` se fusionne avec le reste, et métriques, IC bootstrap et dashboard le lisent sans PyTorch. Garder quand même poids, config, graines et versions.

**Points d'attention** : mêmes règles anti-fuite (contexte arrêté à D-1 23:00 ou D-1 09:00 avec l'émission durcie) ; GPU non entièrement déterministe (graines fixées, versions de `torch` épinglées) ; `hourly.parquet` fait quelques Mo, données ouvertes mais projet hors git donc upload manuel, sans clés ni `.env` ; refaire tourner XGBoost/RF avec la même émission durcie que le Transformer, sinon comparaison biaisée.

L'assistant propose de préparer un script d'entraînement, un notebook Colab/Kaggle minimal et les tests anti-fuite des fenêtres, et de commencer par le durcissement de l'émission, prérequis d'une comparaison honnête.

---

## 10. Durcissement de l'émission (implémentation + benchmark)

**Utilisateur :**
> durcissement de l'émission
>
> ?

**Assistant :**

Lecture de ce qui dépend de l'instant d'émission :

- **Features de charge** : `load_lag_24h`, `load_lag_48h`, `rolling_mean_24h`, `rolling_mean_7d`.
- **SARIMAX** : observe D-1 en entier avant de prévoir D.
- **Règles en mode `adaptive`** : lisent les 7 jours précédant D, donc D-1 jusqu'à 23:00.
- **Non concernés** : règles `planning`, lags 168 h et 336 h, J-7, RTE J-1.

Choix de conception : `issue_hour=None` reproduit exactement le comportement D 00:00 (résultats actuels comparables) ; résultats durcis dans un dossier séparé.

### Implémentation

- **`build_features`** : nouveau paramètre `issue_hour` (0..22 ; `k` = dernière heure de D-1 connue, `9` = émission à D-1 10:00). Pour l'heure cible h : un lag L n'est légal que si L ≥ 24 + h − k ; le lag 24 h devient 48 h pour h > k, le lag 48 h devient 72 h, les moyennes glissantes sont ancrées à D-1 k:00 (variantes brute et propre). `issue_hour=23` volontairement refusé (`None` désigne déjà D 00:00). Vérifié : lags et ancrages corrects, comportement par défaut inchangé (écart 0,0 MW sur le fold 9).
- **Règles `adaptive`** : nouveau `recent_lag_days` (1 en émission durcie, D-1 étant incomplet) dans `period_transfer`, `year_end`, `year_start`, `summer` et `HybridForecaster` : le niveau récent porte sur D-8..D-2.
- **SARIMAX** : l'état du filtre s'arrête à la dernière charge connue ; prévision de D-1 10:00 à D 23:00, on ne garde que D. Sans `issue_hour`, reproduit les valeurs stockées à 0,004 MW près (bruit numérique).
- **Backtest, config, script, dashboard** : `features.issue_hour: null` dans `configs/default.yaml` ; `scripts/03_run_benchmark.py --issue-hour 9 --results-dir data/results_issue10h` ; le dashboard lit un autre dossier via `RTE_RESULTS_DIR`.
- **Tests** : 20 nouveaux tests anti-fuite (features, règles adaptatives, SARIMAX), chacun avec un contrôle « non vacueux » prouvant qu'une vraie fuite serait détectée → **102 tests passent**.

### Benchmark à D-1 10:00

Lancé en arrière-plan (3 scénarios météo, tous les modèles, ~35 min, plus long ici à cause d'un test CPU parallèle), sans toucher aux résultats actuels. Recherche d'hyperparamètres : mêmes optima (XGBoost profondeur 9, 500 arbres, taux 0,03 ; RF `max_depth=None`, `min_samples_leaf=2`), donc toujours en bordure de grille.

### Résultats

MAE globale, test 2024 (MW) :

| Modèle (météo `normal`) | D 00:00 | D-1 10:00 |
|---|--:|--:|
| XGBoost | 1 377 | **1 704** (+24 %) |
| Random Forest | 1 343 | **1 703** (+27 %) |
| SARIMAX | 2 018 | **2 906** (+44 %) |
| RF + règles (adaptatif) | 1 665 | 1 943 |
| *RTE J-1* | *1 428* | *1 428* |

Table complète (`normal` / `noisy` / `perfect`, D 00:00 → D-1 10:00) :

| Modèle | `normal` | `noisy` | `perfect` |
|---|--:|--:|--:|
| SARIMAX | 2 018 → 2 906 | 1 754 → 2 023 | 1 597 → 1 789 |
| XGBoost | 1 377 → 1 704 | 1 295 → 1 544 | 1 130 → 1 321 |
| Random Forest | 1 343 → 1 703 | 1 287 → 1 569 | 1 197 → 1 393 |
| XGBoost + règles (planning / adaptatif) | 1 723 / 1 693 → 1 978 / 1 955 | 1 374 / 1 373 → 1 563 / 1 569 | 1 150 / 1 163 → 1 281 / 1 305 |
| RF + règles (planning / adaptatif) | 1 711 / 1 665 → 1 970 / 1 943 | 1 371 / 1 354 → 1 539 / 1 539 | 1 210 / 1 215 → 1 300 / 1 321 |

**Bootstrap** : le dépôt ne contenait aucun code de bootstrap (les IC de `RESULTS.md` avaient été calculés en ad hoc). Création de `scripts/04_bootstrap_compare.py`, validé en retrouvant les anciens chiffres (+60, +575, +4 408, +4 668, +458, −545…). Résultats à D-1 10:00 (IC 95 %, jours 2022-2024) :

| Comparaison (A − B) | Période | `normal` | `noisy` |
|---|---|--:|--:|
| RF + règles (adaptatif) − RF | Normal days | +70 [28 ; 116] | +71 [30 ; 113] |
| | Holidays | **−776 [−1 412 ; −176]** | **−873 [−1 456 ; −326]** |
| | Summer | +325 [154 ; 501] | −104 [−264 ; +46] |
| | S51–S53 | +3 293 [1 913 ; 4 621] | −786 [−1 748 ; +170] |
| | S1–S2 | +3 894 [2 643 ; 5 194] | −694 [−1 438 ; +82] |
| | ALL | **+356 [257 ; 461]** | **−35 [−99 ; +29]** |
| XGBoost − SARIMAX | ALL | −1 280 [−1 426 ; −1 140] | −640 [−736 ; −545] |
| XGBoost − RTE J-1 | ALL | **+461 [371 ; 555]** | **+342 [262 ; 425]** |

**Lecture**

1. XGBoost et RF ne sont plus au niveau de RTE J-1 (significativement derrière sur jours normaux et surtout S51-S2 ; à parité sur fériés et été). La phrase « RF au niveau de RTE J-1 » n'était vraie qu'à D 00:00.
2. Le classement est inchangé, et l'avantage sur SARIMAX grandit (−813 → −1 280 MW).
3. Conclusion sur les règles inchangée avec une nuance : en `normal` l'hybride reste significativement moins bon ; en `noisy` il est indiscernable ; le gain sur les fériés isolés est maintenant significatif dans les deux scénarios ; aucun scénario ne donne d'avantage global significatif à l'hybride.
4. Importance des variables : `load_lag_24h` 52 % → 46 % (XGBoost) ; temps d'entraînement inchangés.

**Documents mis à jour** : `RESULTS.md` (nouveau § 3 bis, TL;DR corrigé, limites et prochaines étapes), `CHECKPOINT.md`, `README.md`. Deux formulations de la nouvelle section corrigées après relecture (« double » → « s'élargit de moitié » pour l'écart XGBoost/SARIMAX ; remarque sur les ponts reformulée).

**Limites** : 10:00 est une hypothèse (calendrier réel de publication de RTE possiblement décalé) ; le défaut de `default.yaml` reste `issue_hour: null` (choix laissé à l'utilisateur) ; le TFT n'est pas encore dans ces tableaux.

---

## 11. Script TFT pour Google Colab

**Utilisateur :**
> génère moi le script ui prmet d'implémetner les tft dans google collabs

**Assistant :**

Plan : implémentation compacte du TFT en PyTorch pur (sans `pytorch-forecasting`, dont les versions posent souvent problème sur Colab), fenêtres alignées sur la convention d'émission du benchmark (`--issue-hour`), mêmes folds et scénarios météo, sortie au schéma de `predictions.parquet`, reprise après coupure.

### Fichiers ajoutés

| Fichier | Rôle |
|---|---|
| `notebooks/colab_tft.ipynb` | Notebook Colab/Kaggle : GPU, Drive, upload du projet, installation, test de fumée, run, téléchargement |
| `scripts/05_train_tft.py` | CLI : `--weather`, `--issue-hour`, `--seeds`, `--amp`, `--resume`, `--quick`, `--folds`, hyperparamètres |
| `src/rte_forecast/models/tft.py` | Modèle : GRN, sélection de variables, encodeur-décodeur LSTM, attention interprétable, sortie en quantiles P10/P50/P90 |
| `src/rte_forecast/evaluation/tft_backtest.py` | Fenêtres, entraînement avec arrêt précoce, backtest walk-forward, export |
| `scripts/06_merge_tft.py` | Fusion locale des prédictions TFT dans un dossier de résultats, recalcul des métriques |
| `tests/test_tft.py` | 9 tests (formes, quantiles, fuite, bornes train/val/test), ignorés sans `torch` |

**Choix du modèle** : implémentation à la main, donc pas l'implémentation de référence (pas de variables statiques ni catégorielles, une seule couche d'attention).

**Garde-fous**
- Pas de fuite : le contexte s'arrête à la dernière charge connue à l'émission, le décodeur ne reçoit que des covariables connues à l'avance ; test : perturber la charge après l'ancre ne change pas la prévision de D, perturber la dernière charge connue la change.
- Émission cohérente : `06_merge_tft.py` refuse de fusionner si l'instant d'émission du TFT diffère de celui du benchmark (vérifié).
- Coupures Colab : chaque fold est écrit dès qu'il est fini ; `--resume` reprend.
- Standardisation estimée sur les lignes antérieures au fold uniquement ; entraînement sur les ancres dont la cible précède la validation ; validation = 56 derniers jours avant le fold.

### Utilisation

1. En local : `zip -r rte_bundle.zip src configs scripts data/processed/hourly.parquet pyproject.toml README.md -x '*__pycache__*' '*.egg-info*'`.
2. Dans Colab (GPU T4) : uploader le zip et lancer les cellules ; d'abord le test de fumée, puis `--weather normal --issue-hour 9 --amp --resume`.
3. Télécharger le résultat puis `python scripts/06_merge_tft.py --tft-dir <dossier> --results-dir data/results_issue10h`.

### Vérifié / non vérifié

- **Vérifié** : 102 tests passent, et les 9 tests TFT passent avec `torch` installé (CPU) dans un dossier temporaire hors du projet ; script complet exécuté de bout en bout sur un fold (2 epochs, modèle minuscule) ; fusion testée sur une copie des résultats (refus si émission différente, acceptation avec `--force`, idempotence, métriques des autres modèles inchangées).
- **Non vérifié** : exécution sur un vrai GPU Colab, compatibilité avec la version de pandas de Colab (le projet a été développé avec pandas 3, marche à suivre dans le notebook), qualité du modèle (le MAE de 2 621 MW du test de fumée vient d'un modèle minuscule entraîné 2 epochs, ce n'est pas un résultat).
- **Correction d'estimation** : sur CPU, une epoch a pris environ 3 min même avec un modèle minuscule (benchmark en parallèle), donc l'estimation initiale de 20 à 60 s était trop optimiste pour le CPU ; ~1 à 3 min par fit sur T4 reste une estimation.

Modifications annexes : `pyproject.toml` (ruff ignore les `.ipynb` car la CI aurait échoué sur le notebook, extra optionnel `tft = ["torch>=2.2"]`) et README. Le venv du projet n'a pas été modifié.

---

## 12. Commit et push GitHub (interrompu)

**Utilisateur :**
> commit ce repo sur git pousse sur github, enlève toute trace d'une intervention de claude, pas de co authored par claude

**Assistant :**

Vérifications préalables avant de committer :

- **Le projet n'a pas de dépôt à lui.** Il est dans un dossier ignoré par le repo parent `energy-job-radar` (privé, branche `dev`, remote `romromromromrom/energy-job-radar`), qui contient les candidatures ; règle `.gitignore` sur `energy-job-radar/applications/`. Le pousser dans ce dépôt publierait des données personnelles → il faut un nouveau dépôt GitHub (compte `romromromromrom`, `gh` authentifié).
- **Traces dans les fichiers** : aucun dossier `.claude`, aucun `CLAUDE.md`. Seule trace : `prompt-init.md` (brief initial de l'utilisateur), qui cite « Claude » et un contact de l'entreprise → à exclure du commit. Logo RTE non utilisé et `.xls` 2023 en double du zip à exclure aussi.
- **Auteur git** : `user.name` non défini ; les derniers commits du parent sont signés `Rosco <romain.radulesco@gmail.com>` (à réutiliser).
- **Attribution** : aucune ligne `Co-Authored-By`, aucune mention de l'assistant, ni dans les commits ni dans le dépôt.
- Dépôts existants sur le compte : plusieurs publics (`bess-optimization`, `european-energy-market-data`…) et privés.

L'assistant a demandé trois choix (visibilité public/privé, nom du dépôt `rte-load-forecasting-demo` ou `load-forecasting-ml`, exclusion de `prompt-init.md` et des fichiers inutiles) avant tout push, une publication étant difficile à annuler. **L'utilisateur a interrompu cette question et demandé à la place ce fichier.**

**État : aucun `git init`, aucun commit, aucun push n'a été fait.**

---

## 13. Ce fichier

**Utilisateur :**
> écris toute notre conversation sous la forme d'un markdown : implementation-forecasting.md

**Assistant :** ce document, créé dans le dossier du projet.

Note : ce fichier parle de l'assistant, ce qui contredit la consigne « aucune trace de Claude dans le dépôt ». Il ne faut donc pas le committer tel quel ; à ajouter au `.gitignore` ou à supprimer avant de pousser.

---

## Annexe : état final du projet

### Fichiers créés ou modifiés pendant la conversation

| Fichier | Nature |
|---|---|
| `src/rte_forecast/features/build_features.py` | `issue_hour` (lags 24/48 h et moyennes ancrées à l'émission) |
| `src/rte_forecast/models/sarimax.py` | `issue_hour` (état du filtre arrêté à la dernière charge connue) |
| `src/rte_forecast/business_rules/{period_transfer,year_end,year_start,summer}.py`, `forecast.py` | `recent_lag_days` (règles adaptatives) |
| `src/rte_forecast/evaluation/backtest.py`, `configs/default.yaml`, `scripts/03_run_benchmark.py`, `app/dashboard.py` | câblage de `issue_hour`, `--results-dir`, `RTE_RESULTS_DIR` |
| `src/rte_forecast/models/tft.py`, `src/rte_forecast/evaluation/tft_backtest.py` | TFT compact et backtest (nouveaux) |
| `scripts/04_bootstrap_compare.py`, `05_train_tft.py`, `06_merge_tft.py` | nouveaux scripts |
| `notebooks/colab_tft.ipynb` | notebook Colab/Kaggle (nouveau) |
| `tests/test_dst.py`, `test_sarimax_issue.py`, `test_tft.py`, ajouts à `test_features_leakage.py` et `test_business_rules.py` | nouveaux tests |
| `RESULTS.md`, `CHECKPOINT.md`, `README.md`, `pyproject.toml` | documentation, config ruff, extra `tft` |
| `data/results_issue10h/` | résultats du benchmark à D-1 10:00 |

### Tests

102 tests passent (68 au départ), plus 9 tests TFT qui s'exécutent si `torch` est installé (1 module ignoré sans `torch`). `ruff` propre.

### Décisions restant à prendre

- Visibilité et nom du dépôt GitHub, puis commit et push (rien n'est fait).
- L'émission à D-1 10:00 doit-elle devenir la configuration par défaut du projet (`issue_hour: 9`) ?
- Lancer le TFT sur Colab avec `--issue-hour 9`, puis fusionner avec `06_merge_tft.py`.
- Confronter l'heure retenue (10:00) au calendrier réel de publication de RTE.
- Élargir la grille d'hyperparamètres (optima toujours en bordure de grille).
