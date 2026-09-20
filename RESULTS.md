# RESULTS — benchmark de prévision de la consommation française

> Généré à partir de `data/results/*.parquet` (run du 2026-09-19). Tout est reproductible :
> `scripts/01_prepare_data.py → 02_analyze_hypotheses.py → 03_run_benchmark.py`, puis
> `streamlit run app/dashboard.py`. Voir `CHECKPOINT.md` pour les choix et difficultés.

## TL;DR

1. **XGBoost bat SARIMAX nettement** : MAE 1 377 vs 2 018 MW sur le test 2024 (−32 %), 7× plus vite à entraîner, et l'écart est significatif (IC 95 % bootstrap, 2022-2024) sur les jours normaux, fériés, été et S51–S53 ; non significatif sur S1–S2 et les ponts (trop peu de jours).
2. **Le modèle hybride « ML + règles métier » n'est PAS meilleur que le ML seul** dans le scénario opérationnel (`normal`) : RF seul 1 343 MW vs RF + règles 1 665–1 711 MW (test 2024). L'hypothèse de départ est **infirmée en l'état**, et de façon statistiquement solide sur S51–S2 et l'été.
3. Les règles apportent un gain **uniquement sur les fériés isolés** (−545 MW, IC 95 % [−1 073 ; −59] en météo `noisy`) et, avec une météo **parfaite** (oracle), sur S1–S2 (1 806 vs 2 663 MW) ; hors de ces cas elles sont neutres ou pénalisantes.
4. La cause dominante de l'échec des règles n'est pas le calendrier mais **la météo** : avec une météo parfaite, le moteur fin/début d'année tombe à ~1,4–2,6 GW de MAE (vs 4–11 GW pour J-7) ; avec la seule normale de température, l'anomalie météo de l'année cible (hivers 2022 et 2023 très doux) l'emporte.
5. Le meilleur résultat global est celui du **Random Forest / XGBoost « tous jours »** (MAE ≈ 1,35–1,38 GW, MAPE 2,6 %), au niveau de la prévision RTE J-1 (1 428 MW) — **mais avec un avantage d'information** (cf. Limites).

6. **Émission durcie (D-1 10:00, § 3 bis)** : les tableaux des §§ 1-4 supposent une prévision émise à **D 00:00** (charge de la veille complète), ce qui est optimiste. Avec une émission réaliste à D-1 10:00, XGBoost/RF perdent ≈ 24 % (MAE 1 704 / 1 703 MW) et **repassent derrière RTE J-1** (1 428 MW ; écart XGBoost − RTE : +461 MW, IC 95 % [371 ; 555]). Le classement XGBoost ≈ RF ≪ SARIMAX ≪ J-7 est inchangé (XGBoost − SARIMAX : −1 280 MW [−1 426 ; −1 140]) et les conclusions sur les règles tiennent (hybride toujours significativement moins bon en `normal`, indiscernable en `noisy`, gain sur les fériés désormais significatif dans les deux scénarios). **La phrase « RF au niveau de RTE J-1 » du point 5 n'est vraie qu'à D 00:00.**

> ⚠️ Ces chiffres reposent sur **5 ans d'historique** (2020-2024, 2020 = COVID) et sur une prévision de température qui est **la normale** (pas de vraie prévision). Ne pas les sur-interpréter.

## 1. Architecture

```
eCO2mix (5 zips) ─┐                          ┌─ Naive J-7 / RTE J-1 (référence) / SARIMAX
Enedis T° (API) ──┼─► hourly.parquet ─► features ─► XGBoost / RF « tous jours » (drapeaux calendaires)
Open-Meteo (proxy)┘   calendrier          │
                      (NORMAL / HOLIDAY / BRIDGE / SUMMER / YEAR_END / YEAR_START)
                                          └─► XGBoost / RF « jours normaux » ─┐
                                                                               ├─► forecast.py (routeur) ─► prévision + explication
                       règles : year_end · year_start · summer · isolated_holidays ┘
```

Code : `src/rte_forecast/{data,calendar,features,models,business_rules,evaluation}` + `forecast.py` (routeur, aucune règle dedans) ; détail dans `README.md`.

## 2. Données

| Source | Contenu | Statut |
|---|---|---|
| RTE eCO2mix « Annuel-Définitif » 2020–2024 (fichiers fournis) | Consommation (pas 30 min → agrégée à l'heure), prévision J-1 RTE | **Réel**, complet : 43 848 h, 0 doublon, 0 manquant |
| Enedis open data — *température et pseudo-rayonnement* (API `donnees-de-temperature-et-de-pseudo-rayonnement`) | T° réalisée et normale lissées, nationale | **Réel** mais seulement depuis 2021-09-18 (fenêtre glissante de 5 ans) |
| Open-Meteo archive (ERA5), 12 villes pondérées | T° 2020-01 → 2021-09 | **PROXY** recalé sur Enedis (RMSE 1,00 °C, corr. 0,987) — 15 048 h |
| Prévision de température `noisy` | réalisée + erreur AR(1) | **SYNTHÉTIQUE** (RMSE 1,24 °C) |

Découpage : **train ≤ 2022 · validation 2023 · test 2024**, plus 12 folds walk-forward trimestriels (2022-T1 → 2024-T4, entraînement expanding). Prévision du jour D émise à D 00:00 (retards ≥ 24 h).

Jours par régime (2022-2024) : NORMAL 874 · SUMMER 105 · YEAR_END 44 · YEAR_START 44 · HOLIDAY 24 · **BRIDGE 5**. (Test 2024 seul : 292 / 35 / 14 / 16 / 8 / **1**.) Les ponts sont trop rares pour conclure quoi que ce soit.

## 3. Performances — test 2024, scénario opérationnel `normal` (T° prévue = normale)

MAE en MW. Meilleure valeur *hors référence RTE* en gras.

| Période | Naive J-7 | SARIMAX | XGBoost | RF | RF + règles (planning) | RF + règles (adaptatif) | *RTE J-1* |
|---|--:|--:|--:|--:|--:|--:|--:|
| Normal days | 2 835 | 1 916 | 1 318 | **1 273** | 1 330 | 1 330 | *1 414* |
| Holidays | 5 043 | 4 792 | **1 105** | 1 600 | 1 301 | 1 301 | *2 292* |
| Bridge days (n = 1 j) | 8 204 | **3 107** | 2 265 | 2 837 | 3 847 | 3 847 | *1 228* |
| Summer (Aug window) | 2 004 | 1 340 | **699** | 707 | 1 654 | 1 329 | *919* |
| S51–S53 | 4 965 | 2 703 | **2 882** | 2 961 | 3 375 | 3 860 | *1 772* |
| S1–S2 | 10 371 | 3 314 | 2 700 | **2 368** | 7 398 | 6 627 | *2 079* |
| **ALL** | 3 229 | 2 018 | 1 377 | **1 343** | 1 711 | 1 665 | *1 428* |

| Global (test 2024, `normal`) | MAE | RMSE | Biais | MAPE | Fit / fold |
|---|--:|--:|--:|--:|--:|
| Naive J-7 | 3 229 | 4 846 | −163 | 6,05 % | — |
| SARIMAX | 2 018 | 2 732 | +453 | 4,14 % | 67 s |
| XGBoost | 1 377 | 2 040 | +309 | 2,62 % | 9 s |
| Random Forest | **1 343** | 2 000 | +185 | 2,59 % | 35 s |
| XGBoost + règles (planning / adaptatif) | 1 723 / 1 693 | 2 683 / 2 631 | +104 / +246 | 3,26 / 3,20 % | 9 s (+4 s règles) |
| RF + règles (planning / adaptatif) | 1 711 / 1 665 | 2 684 / 2 604 | +52 / +177 | 3,25 / 3,16 % | 29 s (+4 s règles) |
| *RTE J-1 (référence externe)* | 1 428 | 1 766 | −1 146 | 2,86 % | — |

Sensibilité au scénario météo (MAE globale, test 2024) :

| Modèle | `normal` (opérationnel) | `noisy` (synthétique) | `perfect` (oracle) |
|---|--:|--:|--:|
| SARIMAX | 2 018 | 1 754 | 1 597 |
| XGBoost | 1 377 | 1 295 | **1 130** |
| Random Forest | 1 343 | **1 287** | 1 197 |
| XGBoost + règles (planning) | 1 723 | 1 374 | 1 150 |
| RF + règles (adaptatif) | 1 665 | 1 354 | 1 215 |

Le tableau complet par période, pour les 3 scénarios et les 3 splits (train_cv 2022 / validation 2023 / test 2024), se consulte dans le dashboard (onglet « Choix du modèle »).

### Le hybride est-il meilleur que le RF seul ? (test statistique)

Écart de MAE journalière **RF + règles (adaptatif) − RF**, bootstrap apparié sur les jours 2022-2024 (> 0 = l'hybride est *moins bon*) :

| Période | n jours | Scénario `normal` | Scénario `noisy` |
|---|--:|--:|--:|
| Normal days | 874 | +60 [27 ; 97] | +46 [15 ; 80] |
| Holidays | 24 | −454 [−1 014 ; +67] | **−545 [−1 073 ; −59]** |
| Bridge days | 5 | +127 [−1 085 ; +1 148] | +270 [−881 ; +1 389] |
| Summer | 105 | **+575 [415 ; 748]** | +113 [−7 ; 226] |
| S51–S53 | 44 | **+4 408 [3 137 ; 5 669]** | +128 [−710 ; 923] |
| S1–S2 | 44 | **+4 668 [3 469 ; 5 845]** | +126 [−451 ; 687] |
| ALL | 1 096 | **+458 [349 ; 565]** | +47 [−5 ; 96] |

Lecture : avec la seule normale de température, l'hybride est significativement pire sur S51–S2 et l'été. Avec une météo réaliste (`noisy`), il devient **indiscernable** du RF seul (sauf fériés isolés, où il gagne). Il n'y a **aucun scénario** où l'hybride bat significativement le RF seul globalement.

### XGBoost vs SARIMAX

MAE journalière XGBoost − SARIMAX (2022-2024, `normal`) : Normal −782 [−878 ; −683] · Fériés −3 422 [−4 136 ; −2 762] · Été −681 [−810 ; −554] · S51–S53 −1 131 [−1 831 ; −431] · S1–S2 −13 [−559 ; +599] · ALL **−813 [−905 ; −724]**. Sur le seul test 2024, XGBoost est meilleur partout sauf S51–S53 (différence non significative, 14 jours).

Pourquoi : (i) `load_lag_24h` porte à lui seul l'essentiel du signal (SHAP moyen 5,9 GW, 3,7× le 2ᵉ) et le lag 168 h capture la saisonnalité hebdomadaire, ce que SARIMAX (saisonnalité 24 h + 5 indicatrices de jour) modélise mal ; (ii) les fériés sont mal traités par SARIMAX (4 792 vs 1 105 MW) : un indicateur exogène linéaire ne reproduit pas une chute de charge dépendant de l'heure et du jour de semaine ; (iii) SARIMAX est ré-estimé sur une fenêtre glissante de 84 jours (coût), donc voit peu de fériés ; (iv) l'effet température est non linéaire (chauffage) et interagit avec l'heure/le week-end. SARIMAX reste l'ancre méthodologique correcte et bat J-7 (−38 %).

### Importance des variables

* XGBoost : `load_lag_24h` 52 %, `load_lag_168h` 12 %, `rolling_mean_24h` 7 %, `temp_sq_x_weekend` 4 %, `is_weekend` 4 %.
* RF : `load_lag_24h` 53 %, `rolling_mean_24h` 20 %, `load_lag_168h` 12 %.
* SHAP : la **température prévue pèse très peu** en `normal` (la normale ≈ fonction de la saison, donc redondante avec `month`/lags) ; c'est ce qui explique le gain quand on passe à `noisy`/`perfect`.

### Temps d'entraînement (moyenne par fold ; 1 thread par ajustement, 4 ajustements en parallèle sur 8 cœurs ; ≤ 41 k lignes)

XGBoost 9 s · Random Forest 35 s · SARIMAX 67 s (fit seul, fenêtre 84 j) · règles métier ≈ 4 s pour tout le backtest.

### Validation croisée (12 folds, MAE moyenne ± écart-type, `normal`)

RTE J-1 1 281 ± 274 · RF 1 382 ± 545 · XGBoost 1 403 ± 621 · SARIMAX 2 215 ± 641 · RF + règles 1 842 ± 932 · J-7 3 395 ± 1 541. Les folds contenant fin/début d'année (F1, F4, F5, F8, F9, F12) sont parmi les plus dégradés pour tous les modèles ; les règles y sont les plus pénalisées.

## 3 bis. Émission durcie : prévision de D émise à D-1 10:00

`features.issue_hour: 9` (`python scripts/03_run_benchmark.py --issue-hour 9 --results-dir data/results_issue10h`) : la dernière charge connue est D-1 09:00. Conséquences codées (voir `features/build_features.py`) : pour l'heure cible h > 9, le « lag 24 h » devient un lag 48 h (et le « lag 48 h » un lag 72 h) ; les moyennes glissantes 24 h et 7 j sont ancrées à D-1 09:00 ; SARIMAX prévoit de D-1 10:00 à D 23:00 et ne garde que D ; les règles *adaptatives* lisent D-8..D-2 (D-1 est incomplet). Les règles *planning*, J-7, RTE J-1 et les lags 168/336 h ne changent pas. Hyperparamètres : les mêmes optima sont retenus (toujours en bordure de grille). Résultats : `data/results_issue10h/`. Reproductible : `scripts/04_bootstrap_compare.py` (IC bootstrap) ; 20 tests anti-fuite dédiés (`tests/test_features_leakage.py`, `test_business_rules.py`, `test_sarimax_issue.py`).

MAE globale, test 2024 (MW) :

| Modèle | `normal` D 00:00 | `normal` **D-1 10:00** | `noisy` D 00:00 | `noisy` **D-1 10:00** | `perfect` D 00:00 | `perfect` **D-1 10:00** |
|---|--:|--:|--:|--:|--:|--:|
| SARIMAX | 2 018 | **2 906** | 1 754 | 2 023 | 1 597 | 1 789 |
| XGBoost | 1 377 | **1 704** | 1 295 | 1 544 | 1 130 | 1 321 |
| Random Forest | 1 343 | **1 703** | 1 287 | 1 569 | 1 197 | 1 393 |
| XGBoost + règles (planning / adaptatif) | 1 723 / 1 693 | 1 978 / 1 955 | 1 374 / 1 373 | 1 563 / 1 569 | 1 150 / 1 163 | 1 281 / 1 305 |
| RF + règles (planning / adaptatif) | 1 711 / 1 665 | 1 970 / 1 943 | 1 371 / 1 354 | 1 539 / 1 539 | 1 210 / 1 215 | 1 300 / 1 321 |
| *RTE J-1* (inchangé) | *1 428* | *1 428* | | | | |
| *J-7* (inchangé) | *3 229* | *3 229* | | | | |

Par période, `normal`, D-1 10:00 (MAE, MW) :

| Période | J-7 | SARIMAX | XGBoost | RF | RF + règles (planning) | RF + règles (adaptatif) | *RTE J-1* |
|---|--:|--:|--:|--:|--:|--:|--:|
| Normal days | 2 835 | 2 864 | 1 642 | 1 607 | 1 658 | 1 658 | *1 414* |
| Holidays | 5 043 | 6 030 | 1 438 | 1 943 | **1 191** | **1 191** | *2 292* |
| Bridge days (n = 1 j) | 8 204 | 6 789 | 2 192 | 3 286 | 3 906 | 3 906 | *1 228* |
| Summer | 2 004 | 1 499 | **863** | 958 | 1 654 | 1 345 | *919* |
| S51–S53 | 4 965 | 3 435 | 3 507 | 3 733 | 3 375 | 3 771 | *1 772* |
| S1–S2 | 10 371 | 4 471 | 3 206 | **3 094** | 7 398 | 7 116 | *2 079* |
| **ALL** | 3 229 | 2 906 | 1 704 | 1 703 | 1 970 | 1 943 | *1 428* |

Écarts de MAE journalière, bootstrap apparié (`scripts/04_bootstrap_compare.py`, jours 2022-2024, IC 95 %) :

| Comparaison (A − B) | Période | `normal` | `noisy` |
|---|---|--:|--:|
| RF + règles (adaptatif) − RF | Normal days | +70 [28 ; 116] | +71 [30 ; 113] |
| | **Holidays** | **−776 [−1 412 ; −176]** | **−873 [−1 456 ; −326]** |
| | Summer | +325 [154 ; 501] | −104 [−264 ; +46] |
| | S51–S53 | +3 293 [1 913 ; 4 621] | −786 [−1 748 ; +170] |
| | S1–S2 | +3 894 [2 643 ; 5 194] | −694 [−1 438 ; +82] |
| | **ALL** | **+356 [257 ; 461]** | **−35 [−99 ; +29]** |
| XGBoost − SARIMAX | ALL | −1 280 [−1 426 ; −1 140] | −640 [−736 ; −545] |
| XGBoost − RTE J-1 | Normal days | +382 [288 ; 479] | +260 [179 ; 345] |
| | S51–S53 / S1–S2 | +1 941 / +2 098 (signif.) | +1 777 / +1 774 (signif.) |
| | Holidays / Summer | −215 / −35 (non signif.) | −143 / −79 (non signif.) |
| | **ALL** | **+461 [371 ; 555]** | **+342 [262 ; 425]** |

Lecture :

1. **Le coût de l'émission réaliste est de ≈ 19-27 % pour les modèles ML** (XGBoost +327 MW, RF +360 MW en `normal`), concentré sur les heures cibles 10-23 h (MAE ≈ 1,3-1,4 GW → 1,8-2,0 GW, cf. audit) ; SARIMAX perd davantage (+44 %), donc son écart avec XGBoost **s'élargit de moitié** (−813 → −1 280 MW).
2. **XGBoost/RF ne sont plus au niveau de RTE J-1** : significativement derrière sur les jours normaux et surtout S51–S2 (RTE dispose de vraies prévisions météo et d'un savoir-faire calendaire), à parité sur fériés et été. Avec D 00:00 l'écart était de −51 MW (`normal`) : il était en grande partie un artefact d'information.
3. **Conclusions sur les règles inchangées, une nuance en plus** : en `normal` l'hybride reste significativement moins bon (S51–S2 et été) ; en `noisy` il devient indiscernable du ML (−35 [−99 ; +29]) ; le gain sur les fériés isolés (24 jours) est maintenant significatif dans **les deux** scénarios ; les ponts (5 jours) restent trop rares pour conclure. En `noisy`, les règles gagnent en valeur ponctuelle sur S51–S2 mais l'IC contient 0 (44 jours chacun). Aucun scénario ne donne d'avantage significatif global à l'hybride.
4. L'importance des variables se déplace légèrement (XGBoost : `load_lag_24h` 52 % → 46 %, `rolling_mean_24h` 7 % → 8 %) ; temps d'entraînement inchangés.

Hypothèses de cette section : « D-1 10:00 » = charge connue jusqu'à D-1 09:00 (le calendrier réel de publication de RTE peut décaler l'écart) ; les statistiques de fériés isolés utilisent encore des journées entières antérieures à D (au plus une demi-journée d'optimisme) ; SARIMAX est ré-estimé au début du fold sur des données jusqu'à D0-1 23:00 (quelques heures d'avance sur l'état du filtre, effet négligeable).

## 4. Principaux enseignements

1. **Le lag 24 h domine le signal** (~52 % de l'importance XGBoost/RF, SHAP 3,7× celui de la 2ᵉ variable) ; XGBoost et RF sont interchangeables (écart 2,5 %), XGBoost beaucoup plus rapide. On n'a **aucune raison technique d'ajouter du deep learning** : un modèle simple bat J-7 de 57 % et SARIMAX de 32 %.
2. **Exclure les jours spéciaux de l'apprentissage n'aide pas** ici : le modèle « jours normaux » fait un peu moins bien sur les jours normaux (1 330 vs 1 273 MW, écart significatif mais faible) que le modèle « tous jours ». Avec des drapeaux calendaires + lags, un arbre n'est pas « contaminé » par les fériés ; il a surtout moins de données quand on les retire. L'argument de contamination de la spec ne se vérifie pas sur ce jeu de données.
3. **Ce n'est pas le calendrier qui bat les règles, c'est la météo.** Sans normalisation météo, le transfert d'une année analogue importait l'anomalie de température (S51 2022 = 0,81 de la baseline, 2023 = 0,96). Avec normalisation météo et météo parfaite, le moteur fin/début d'année est **2 à 6× meilleur que J-7** (1,4–2,6 GW vs 4–11 GW) : la logique niveau → semaine → jour → profil est saine. Elle est pénalisée par (a) la température prévue = normale, (b) un horizon *planning* de 1–3 semaines face à un ML *day-ahead* qui lit la charge de la veille, (c) des années analogues médiocres (scores 0,5–0,75 sur 5 ans d'historique, dont l'année COVID et la crise énergétique).
4. **Les fériés isolés sont le seul vrai succès des règles** (RF+règles 1 301 vs RF 1 600 MW en test 2024 ; −545 MW significatif en `noisy`). Le coefficient est *mesuré* (médiane des ratios historiques à même jour de semaine), pas inventé. XGBoost « tous jours » reste néanmoins meilleur en `normal` (1 105 MW).
5. **Hypothèses de la spec contredites ou nuancées** (détail dans `CHECKPOINT.md § 3`) : creux estival vérifié seulement 2 années sur 5 exactement (±1 semaine en 2021-22, faux en 2020 juin–sept.) ; pas de rampe estivale progressive (creux de ~3 semaines, −6 %) ; Noël et Nouvel An tombent toujours le même jour de semaine.
6. **Comparer aux prévisions RTE J-1** est instructif mais inéquitable : RTE dispose de vraies prévisions météo (donc meilleure sur S51–S2 : 1,8–2,1 GW) ; notre RF utilise la charge de D-1 jusqu'à 23:00, qu'RTE n'a pas à l'émission de sa J-1.

## 5. Limites

* **Instant d'émission** : les §§ 1-4 (D 00:00, charge de la veille complète) sont optimistes ; le durcissement à D-1 10:00 est fait et chiffré au § 3 bis (−24 % de précision pour le ML, RTE J-1 repasse devant). L'heure exacte de publication RTE reste une hypothèse.
* **Prévision de température = normale** : pas une vraie prévision. Le scénario `noisy` est synthétique ; `perfect` est un oracle. Aucune conclusion opérationnelle ne doit s'appuyer sur `perfect`.
* **Historique court** (5 ans, COVID en 2020, crise énergétique 2022-23) : 1 seule année analogue utilisée, souvent médiocre ; 44 jours S51–S53, 44 jours S1–S2, 24 fériés, **5 ponts** sur la période d'évaluation → puissance statistique faible hors normal/été.
* **Température 2020-01 → 2021-09 = proxy** (RMSE 1 °C) ; le hyper-paramétrage et les folds 2022 en dépendent légèrement.
* **Comparaison à information inégale** entre règles *planning* (spec) et ML *day-ahead* ; la variante *adaptative* la réduit mais ne l'élimine pas (elle ne réutilise pas la météo récente ni des lags fins).
* **Hyperparamètres en bordure de grille** (XGBoost profondeur 9 / 500 arbres ; RF `min_samples_leaf`=2) : la grille doit être élargie ; hyperparamètres choisis sur 2023 puis réutilisés en 2022 (léger biais optimiste en 2022).
* **Fallback explicite** : en mode adaptatif, les 30-31 décembre 2024 (semaine ISO 1 de 2025, dont les données ne sont pas disponibles) retombent sur le modèle normal ; ils sont marqués `FALLBACK_NORMAL_MODEL` dans les prédictions.
* **Imputation** jamais exercée sur les vraies données (0 trou) ; validée seulement par injection (MAPE 1,2 % / 3,5 % / 4,4 % pour 3 / 24 / 72 h).
* Le dashboard n'a pas pu être contrôlé visuellement dans un navigateur (MCP Playwright indisponible) ; il est validé en headless (aucune exception sur toutes les interactions testées).

## 6. Prochaines étapes (par ordre de valeur)

1. **Vraies prévisions météo archivées** (Open-Meteo historical forecast, AROME/ARPEGE, ECMWF) pour remplacer `normal`/`noisy` : c'est le facteur qui change le plus les résultats, y compris la conclusion sur les règles.
2. ~~Durcir l'instant d'émission~~ : fait (§ 3 bis). Reste à confronter l'heure retenue (10:00) au calendrier réel de publication RTE.
3. **Allonger l'historique** : eCO2mix définitif 2012-2019 (+ Enedis/ERA5 pour la température) → 8 années de plus pour trouver de meilleurs analogues et estimer les coefficients de fériés (aujourd'hui 24 fériés).
4. **Décision go/no-go sur les règles** : les conserver seulement sur les fériés isolés (gain démontré), puis re-tester S51–S2 et l'été avec des analogues moyennés (top-k pondérés par le score) plutôt qu'un seul.
5. Élargir la grille d'hyperparamètres, ajouter des intervalles de prédiction (quantiles), essayer LightGBM/quantile regression.
6. Seulement ensuite : Airflow, Docker, MLflow, CI/CD, monitoring (Prometheus/Grafana) — et un modèle profond **uniquement si** son gain hors échantillon justifie sa complexité opérationnelle : à ce stade, XGBoost bat déjà nettement les baselines classiques.
