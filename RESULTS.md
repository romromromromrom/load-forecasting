# RESULTS — benchmark de prévision de la consommation française

> Généré à partir de `data/results/*.parquet` (run du 2026-09-20, 72 min sur 1 cœur). Tout est
> reproductible : `scripts/02_analyze_hypotheses.py → 03_run_benchmark.py --save-models`, puis
> `streamlit run app/dashboard.py`. Le run a été comparé au précédent (calculé avec 4 processus) :
> **écart maximal 0,0 MW** sur les 604 992 prédictions. Voir `CHECKPOINT.md` pour les choix et difficultés.
>
> **Convention d'émission unique : la prévision du jour D est émise à D-1 10:00** (charge connue jusqu'à
> D-1 09:00, `features.issue_hour: 9`), comme une vraie prévision J-1. Une première version supposait une
> émission à D 00:00 (charge de la veille complète) : elle était optimiste (≈ −24 % de précision pour les
> modèles ML) et a été abandonnée.

## TL;DR

1. **XGBoost et Random Forest battent SARIMAX nettement** : MAE 1 704 / 1 703 contre 2 906 MW sur le test 2024 (−41 %), et 47 % sous J-7 (3 229 MW). XGBoost s'entraîne 8× plus vite que SARIMAX (5 s contre 43 s par fold). L'écart XGBoost − SARIMAX est significatif (IC 95 % bootstrap, 2022-2024) sur les jours normaux, fériés, été et S51–S53 : −1 280 MW au global [−1 426 ; −1 140] ; non significatif sur S1–S2 et les ponts.
2. **Le modèle hybride « ML + règles métier » n'est PAS meilleur que le ML seul** avec la météo opérationnelle (`normal`) : RF 1 703 MW contre RF + règles 1 943–1 970 MW (écart +356 MW [257 ; 461]). Avec une météo réaliste mais synthétique (`noisy`) il est indiscernable du RF (−35 MW [−99 ; +29]). Aucun scénario *non-oracle* ne donne d'avantage global significatif à l'hybride ; seul l'oracle météo `perfect` (température réalisée, non opérationnel) le fait : RF + règles (adaptatif) − RF = −75 MW [−131 ; −18], dont −1,3 GW en S51–S53.
3. **Les règles ne gagnent que sur les fériés isolés** : −776 MW [−1 412 ; −176] (`normal`) et −873 MW [−1 456 ; −326] (`noisy`) contre le RF, significatif dans les deux scénarios. Contre XGBoost, le gain sur les fériés n'est pas significatif (XGBoost est déjà meilleur que le RF sur les fériés : 1 438 contre 1 943 MW).
4. **Le problème des règles est le niveau, pas la forme** (§ 3 bis) : leur forme horaire est meilleure que celle du RF en fin/début d'année (1 250–1 289 contre 1 546–1 660 MW), mais leur niveau est biaisé de +4 à +6 GW car l'erreur suit l'anomalie de température réelle (1,8–2,2 GW par °C, corrélation 0,86–0,98) que le RF, lui, lit dans la charge de la veille. Avec l'anomalie réelle (oracle) les règles battent le RF (2 187 contre 3 915 MW en S51–S53).
5. **RTE J-1 reste devant nos modèles ML** : XGBoost − RTE = +461 MW [371 ; 555] en `normal`, +342 MW [262 ; 425] en `noisy`. RTE est meilleur sur les jours normaux (+382 MW) et surtout en S51–S2 (+1,9 à +2,1 GW) — il dispose de vraies prévisions météo — et à parité sur fériés et été.

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

## 2. Données et protocole

| Source | Contenu | Statut |
|---|---|---|
| RTE eCO2mix « Annuel-Définitif » 2020–2024 (fichiers fournis) | Consommation (pas 30 min → agrégée à l'heure), prévision J-1 RTE | **Réel**, complet : 43 848 h, 0 doublon, 0 manquant |
| Enedis open data — *température et pseudo-rayonnement* (API `donnees-de-temperature-et-de-pseudo-rayonnement`) | T° réalisée et normale lissées, nationale | **Réel** mais seulement depuis 2021-09-18 (fenêtre glissante de 5 ans) |
| Open-Meteo archive (ERA5), 12 villes pondérées | T° 2020-01 → 2021-09 | **PROXY** recalé sur Enedis (RMSE 1,00 °C, corr. 0,987) — 15 048 h |
| Prévision de température `noisy` | réalisée + erreur AR(1) | **SYNTHÉTIQUE** (RMSE 1,24 °C) |

Découpage : **train ≤ 2022 · validation 2023 · test 2024**, plus 12 folds walk-forward trimestriels (2022-T1 → 2024-T4, entraînement expanding).

**Convention d'émission (D-1 10:00).** Dernière charge connue : D-1 09:00. Pour l'heure cible h, un retard de charge L n'est légal que si L ≥ 24 + h − 9 : le « lag 24 h » devient un lag de 48 h pour h > 9 (et le « lag 48 h » un lag de 72 h), les moyennes glissantes 24 h et 7 j sont ancrées à D-1 09:00 (`features/build_features.py`). SARIMAX prévoit de D-1 10:00 à D 23:00 et ne garde que D ; les règles *adaptatives* lisent D-8..D-2 (D-1 est incomplet) ; les règles *planning*, J-7, RTE J-1 et les lags 168/336 h ne changent pas. 20 tests anti-fuite dédiés. L'heure exacte de publication de RTE reste une hypothèse.

Jours par régime (2022-2024) : NORMAL 874 · SUMMER 105 · YEAR_END 44 · YEAR_START 44 · HOLIDAY 24 · **BRIDGE 5**. (Test 2024 seul : 292 / 35 / 14 / 16 / 8 / **1**.) Les ponts sont trop rares pour conclure quoi que ce soit.

## 3. Performances — test 2024, scénario opérationnel `normal` (T° prévue = normale)

MAE en MW. Meilleure valeur *hors référence RTE* en gras.

| Période | Naive J-7 | SARIMAX | XGBoost | RF | RF + règles (planning) | RF + règles (adaptatif) | *RTE J-1* |
|---|--:|--:|--:|--:|--:|--:|--:|
| Normal days | 2 835 | 2 864 | 1 642 | **1 607** | 1 658 | 1 658 | *1 414* |
| Holidays | 5 043 | 6 030 | 1 438 | 1 943 | **1 191** | **1 191** | *2 292* |
| Bridge days (n = 1 j) | 8 204 | 6 789 | **2 192** | 3 286 | 3 906 | 3 906 | *1 228* |
| Summer (Aug window) | 2 004 | 1 499 | **863** | 958 | 1 654 | 1 345 | *919* |
| S51–S53 | 4 965 | 3 435 | 3 507 | 3 733 | **3 375** | 3 771 | *1 772* |
| S1–S2 | 10 371 | 4 471 | 3 206 | **3 094** | 7 398 | 7 116 | *2 079* |
| **ALL** | 3 229 | 2 906 | 1 704 | **1 703** | 1 970 | 1 943 | *1 428* |

| Global (test 2024, `normal`) | MAE | RMSE | Biais | MAPE | Fit / fold |
|---|--:|--:|--:|--:|--:|
| Naive J-7 | 3 229 | 4 846 | −163 | 6,05 % | — |
| SARIMAX | 2 906 | 3 825 | +1 014 | 5,97 % | 43 s |
| XGBoost | **1 704** | 2 548 | +314 | 3,21 % | 5 s |
| Random Forest | **1 703** | 2 576 | +267 | 3,22 % | 20 s |
| XGBoost + règles (planning / adaptatif) | 1 978 / 1 955 | 2 977 / 2 950 | +148 / +279 | 3,73 / 3,67 % | 6 s (+4 s règles) |
| RF + règles (planning / adaptatif) | 1 970 / 1 943 | 2 969 / 2 932 | +123 / +251 | 3,72 / 3,66 % | 17 s (+4 s règles) |
| *RTE J-1 (référence externe)* | 1 428 | 1 766 | −1 146 | 2,86 % | — |

Sensibilité au scénario météo (MAE globale, test 2024) :

| Modèle | `normal` (opérationnel) | `noisy` (synthétique) | `perfect` (oracle) |
|---|--:|--:|--:|
| SARIMAX | 2 906 | 2 023 | 1 789 |
| XGBoost | 1 704 | 1 544 | **1 321** |
| Random Forest | 1 703 | 1 569 | 1 393 |
| XGBoost + règles (planning) | 1 978 | 1 563 | **1 281** |
| RF + règles (adaptatif) | 1 943 | **1 539** | 1 321 |

Le tableau complet par période, pour les 3 scénarios et les 3 splits (train_cv 2022 / validation 2023 / test 2024), se consulte dans le dashboard (onglet « Choix du modèle »). Validation 2023 (`normal`, MAE) : RTE 1 342 · XGBoost 1 661 · RF 1 735 · RF + règles (adaptatif) 2 232 · SARIMAX 3 723 · J-7 3 446.

### Le hybride est-il meilleur que le RF seul ? (test statistique)

Écart de MAE journalière **RF + règles (adaptatif) − RF**, bootstrap apparié sur les jours 2022-2024 (`scripts/04_bootstrap_compare.py` ; > 0 = l'hybride est *moins bon*) :

| Période | n jours | Scénario `normal` | Scénario `noisy` |
|---|--:|--:|--:|
| Normal days | 874 | +70 [28 ; 116] | +71 [30 ; 113] |
| Holidays | 24 | **−776 [−1 412 ; −176]** | **−873 [−1 456 ; −326]** |
| Bridge days | 5 | −618 [−1 415 ; +210] | −775 [−1 522 ; −27] |
| Summer | 105 | **+325 [154 ; 501]** | −104 [−264 ; +46] |
| S51–S53 | 44 | **+3 293 [1 913 ; 4 621]** | −786 [−1 748 ; +170] |
| S1–S2 | 44 | **+3 894 [2 643 ; 5 194]** | −694 [−1 438 ; +82] |
| ALL | 1 096 | **+356 [257 ; 461]** | −35 [−99 ; +29] |

Avec XGBoost comme modèle « normal » : ALL +395 [296 ; 500] (`normal`) et +32 [−31 ; +94] (`noisy`) ; Holidays +23 [−421 ; +481] et −189 [−622 ; +225] (non significatifs : XGBoost gère déjà bien les fériés).

Lecture : avec la seule normale de température, l'hybride est significativement pire sur S51–S2 et l'été. Avec une météo réaliste (`noisy`), il devient **indiscernable** du RF ; en valeur ponctuelle il gagne même sur S51–S2 mais les intervalles contiennent 0 (44 jours chacun). Le seul gain solide est celui des fériés isolés. Il n'y a **aucun scénario non-oracle** où l'hybride bat significativement le RF seul globalement. Avec l'oracle `perfect` (température réalisée, jamais opérationnelle) l'hybride devient meilleur : RF + règles (adaptatif) − RF = **−75 [−131 ; −18]** globalement, Holidays −682 [−1 241 ; −155], Summer −229 [−403 ; −64], S51–S53 −1 303 [−2 099 ; −529], S1–S2 −1 153 [−1 701 ; −617] ; avec XGBoost, ALL +29 [−28 ; +86] (seul S51–S53 gagne : −1 018 [−1 834 ; −222]). C'est cohérent avec § 3 bis : les règles deviennent utiles dès que la météo est connue.

### XGBoost vs SARIMAX

MAE journalière XGBoost − SARIMAX (2022-2024, `normal`) : Normal −1 342 [−1 497 ; −1 189] · Fériés −2 960 [−3 907 ; −2 062] · Été −804 [−946 ; −653] · S51–S53 −1 418 [−2 433 ; −403] · S1–S2 −26 [−791 ; +706] · Ponts −2 258 [−5 117 ; +122] · ALL **−1 280 [−1 426 ; −1 140]** (`noisy` : ALL −640 [−736 ; −545]). Sur le seul test 2024, XGBoost est meilleur partout sauf S51–S53 (3 507 contre 3 435 MW, 14 jours, non significatif).

Pourquoi : (i) `load_lag_24h` porte à lui seul l'essentiel du signal (SHAP moyen 4,8 GW, 2,5× le 2ᵉ) et le lag 168 h capture la saisonnalité hebdomadaire, ce que SARIMAX (saisonnalité 24 h + indicatrices de jour) modélise mal ; (ii) les fériés sont mal traités par SARIMAX (6 030 contre 1 438 MW) : un indicateur exogène linéaire ne reproduit pas une chute de charge dépendant de l'heure et du jour de semaine ; (iii) SARIMAX est ré-estimé sur une fenêtre glissante de 84 jours, donc voit peu de fériés ; (iv) l'effet température est non linéaire (chauffage) et interagit avec l'heure/le week-end ; (v) avec l'émission à D-1 10:00 SARIMAX doit prolonger son filtre de 14 à 38 h sans observer la charge, ce qui le pénalise davantage que les modèles à retards (+44 % d'erreur contre +24 %). SARIMAX bat J-7 de 10 % seulement, et reste instable d'un fold à l'autre (pire que J-7 sur 5 folds sur 12, dont F6 et F10 : 5 840 et 4 688 MW).

### Importance des variables

* XGBoost : `load_lag_24h` 46 %, `load_lag_168h` 13 %, `rolling_mean_24h` 8 %, `temp_sq_x_weekend` 5 %, `is_weekend` 5 %.
* RF : `load_lag_24h` 47 %, `rolling_mean_24h` 22 %, `load_lag_168h` 16 %, `load_lag_336h` 4 %, `weekday` 2 %.
* SHAP (XGBoost) : `load_lag_24h` 4 843 MW, `load_lag_168h` 1 971, `rolling_mean_24h` 1 414 ; la **température prévue pèse très peu** en `normal` (11ᵉ rang, 190 MW : la normale ≈ fonction de la saison, donc redondante avec `month`/lags) ; c'est ce qui explique le gain quand on passe à `noisy`/`perfect`.

### Temps d'entraînement (moyenne par fold ; 1 processus, 1 thread ; ≤ 43 500 lignes)

XGBoost 5 s · Random Forest 20 s · SARIMAX 43 s (fit seul, fenêtre 84 j) · règles métier ≈ 4 s pour tout le backtest. Benchmark complet (3 scénarios, hyperparamètres, SHAP, sauvegarde) : 72 min sur un seul cœur.

### Validation croisée (12 folds, MAE moyenne ± écart-type, `normal`)

RTE J-1 1 281 ± 274 · XGBoost 1 744 ± 784 · RF 1 772 ± 725 · RF + règles (adaptatif) 2 129 ± 1 070 · SARIMAX 3 024 ± 1 341 · J-7 3 395 ± 1 541. Les folds contenant fin/début d'année (F1, F4, F5, F8, F9, F12) sont les plus dégradés pour XGBoost et le RF (1,9 à 2,8 GW) ; les règles y sont les plus pénalisées (jusqu'à 4,1 GW en F5).

## 3 bis. Pourquoi les règles perdent en S51–S2 : le niveau, pas la forme

Diagnostic sur les prédictions 2022-2024 (`normal`, 44 jours par période), erreur décomposée en niveau de la journée et forme intra-journalière :

| Période | Modèle | MAE | \|Erreur de niveau\| | Biais de niveau | Erreur de forme |
|---|---|--:|--:|--:|--:|
| S51–S53 | RF | 3 915 | 3 690 | +2 143 | 1 546 |
| | Règles (planning) | 6 601 | 6 484 | **+5 923** | **1 289** |
| S1–S2 | RF | 3 518 | 3 139 | −772 | 1 660 |
| | Règles (planning) | 7 768 | 7 679 | **+4 867** | **1 250** |
| Été | RF | 1 063 | 950 | +149 | 593 |
| | Règles (planning) | 1 385 | 1 262 | −925 | 670 |

* **La forme des règles est meilleure que celle du RF en fin/début d'année, mais le niveau est très biaisé.** Régression de l'erreur de niveau journalière sur l'anomalie de température réalisée (réalisée − normale) : pente **1 862 MW/°C en S51–S53** (corr. 0,87) et **1 992 MW/°C en S1–S2** (corr. 0,92) pour les règles planning — c'est la sensibilité thermique elle-même ; le RF, qui lit la charge de la veille et donc le régime météo, est bien moins sensible (685 et 278 MW/°C, corr. 0,36 et 0,24). Ces périodes étaient plus douces que la normale (S51–S53 : +4,9 / +3,7 / +1,5 °C en 2022 / 2023 / 2024 ; S1–S2 : +0,8 / +4,9 / +0,1 °C).
* **Le mode adaptatif ne corrige pas ce défaut** (pentes 1 818 et 2 184 MW/°C) : d'après le code, la normalisation météo retire l'anomalie des charges récentes puis rajoute l'anomalie *prévue*, nulle quand la prévision est la normale ; il jette précisément l'information qui persiste d'un jour à l'autre.
* **Simulation** (hors pipeline, sensibilité fixée à 2 000 MW/°C, une seule fenêtre de 7 jours non ajustée, correction uniforme sur 24 h) :

| Période | RF | Règles planning / adaptatif (actuel) | + persistance de l'anomalie (moyenne D-8..D-2) | + anomalie réelle du jour (oracle) |
|---|--:|--:|--:|--:|
| S51–S53 | 3 915 | 6 601 / 7 208 | 4 756 / 4 814 | **2 187 / 2 220** |
| S1–S2 | 3 518 | 7 768 / 7 478 | 4 316 / 4 939 | **2 596 / 1 656** |

  La persistance simple comble environ 65 à 80 % de l'écart avec le RF mais reste derrière ; avec la météo réelle (oracle, non atteignable) les règles battent nettement le RF. Combiner le niveau du RF avec la forme des règles donne à peu près le RF (S51–S53 3 927 contre 3 915 ; S1–S2 3 366 contre 3 518 ; été 1 095 contre 1 063) : la valeur potentielle est dans le niveau, pas dans la forme.
* **L'été est un autre problème** (biais −925 MW, forme moins bonne : 670 contre 593 MW) : non investigué.

## 4. Principaux enseignements

1. **Le lag 24 h domine le signal** (46–47 % de l'importance XGBoost/RF, SHAP 2,5× celui de la 2ᵉ variable) ; XGBoost et RF sont interchangeables (écart < 0,1 %), XGBoost beaucoup plus rapide. On n'a **aucune raison technique d'ajouter du deep learning** : un modèle simple bat J-7 de 47 % et SARIMAX de 41 %.
2. **Exclure les jours spéciaux de l'apprentissage n'aide pas** ici : le modèle « jours normaux » fait un peu moins bien sur les jours normaux (1 658 contre 1 607 MW pour le RF, écart significatif mais faible : +70 MW [28 ; 116]) que le modèle « tous jours ». Avec des drapeaux calendaires + lags, un arbre n'est pas « contaminé » par les fériés ; il a surtout moins de données quand on les retire.
3. **Ce n'est pas le calendrier qui bat les règles, c'est la météo** (§ 3 bis). Leur logique niveau → semaine → jour → profil est saine (meilleure forme que le RF ; avec une météo parfaite, MAE 3 à 5× plus faible que J-7 en S51–S2 : 2 186 / 2 591 contre 7 540 / 8 073 MW en planning, 2 178 / 1 614 en adaptatif). Elles sont pénalisées par (a) la température prévue = normale, alors que le ML lit le régime météo dans la charge de la veille, (b) un horizon *planning* de 1–3 semaines face à un ML *day-ahead*, (c) des années analogues médiocres (scores 0,5–0,75 sur 5 ans, dont l'année COVID et la crise énergétique).
4. **Les fériés isolés sont le seul vrai succès des règles** (RF + règles 1 191 contre 1 943 MW en test 2024 ; −776 MW [−1 412 ; −176] sur 2022-2024). Le coefficient est *mesuré* (médiane des ratios historiques à même jour de semaine), pas inventé. Mais XGBoost « tous jours » (1 438 MW) fait déjà presque aussi bien, et l'hybride XGBoost n'apporte rien de significatif sur les fériés.
5. **Hypothèses de la spec contredites ou nuancées** (détail dans `CHECKPOINT.md § 3`) : creux estival vérifié seulement 2 années sur 5 exactement (±1 semaine en 2021-22, faux en 2020 juin–sept.) ; pas de rampe estivale progressive (creux de ~3 semaines, −6 %) ; Noël et Nouvel An tombent toujours le même jour de semaine.
6. **Comparer aux prévisions RTE J-1** : RTE reste devant sur les jours normaux et S51–S2 parce qu'il dispose de vraies prévisions météo ; à parité sur les fériés et l'été. Notre ML est battu de 19 % (`normal`) ; l'écart se resserre en `noisy` (+342 MW) et disparaît en `perfect` (oracle).
7. **L'instant d'émission compte beaucoup** : passer de D 00:00 à D-1 10:00 coûte ≈ 24 % de précision aux modèles ML et 44 % à SARIMAX. Toute comparaison à une prévision opérationnelle doit se faire à l'heure d'émission réelle.

## 5. Limites

* **Instant d'émission** : D-1 10:00 est une hypothèse (calendrier réel de publication RTE possiblement différent) ; les statistiques de fériés isolés utilisent des journées entières antérieures à D (au plus une demi-journée d'optimisme) ; SARIMAX est ré-estimé au début du fold sur des données jusqu'à D0-1 23:00 (quelques heures d'avance sur l'état du filtre, effet négligeable).
* **Prévision de température = normale** : pas une vraie prévision. Le scénario `noisy` est synthétique ; `perfect` est un oracle. Aucune conclusion opérationnelle ne doit s'appuyer sur `perfect`.
* **Historique court** (5 ans, COVID en 2020, crise énergétique 2022-23) : 1 seule année analogue utilisée, souvent médiocre ; 44 jours S51–S53, 44 jours S1–S2, 24 fériés, **5 ponts** sur la période d'évaluation → puissance statistique faible hors normal/été.
* **Température 2020-01 → 2021-09 = proxy** (RMSE 1 °C) ; le hyper-paramétrage et les folds 2022 en dépendent légèrement.
* **Comparaison à information inégale** entre règles *planning* (spec) et ML *day-ahead* ; la variante *adaptative* la réduit mais ne l'élimine pas (elle retire l'anomalie météo récente au lieu de la prolonger, cf. § 3 bis). La simulation de persistance du § 3 bis n'est pas intégrée au pipeline.
* **Hyperparamètres en bordure de grille** (XGBoost profondeur 9 / 500 arbres ; RF `min_samples_leaf`=2) : la grille doit être élargie ; hyperparamètres choisis sur 2023 puis réutilisés en 2022 (léger biais optimiste en 2022).
* **Fallback explicite** : en mode adaptatif, les 30-31 décembre 2024 (semaine ISO 1 de 2025, dont les données ne sont pas disponibles) retombent sur le modèle normal ; ils sont marqués `FALLBACK_NORMAL_MODEL` dans les prédictions.
* **Imputation** jamais exercée sur les vraies données (0 trou) ; validée seulement par injection (MAPE 1,2 % / 3,5 % / 4,4 % pour 3 / 24 / 72 h).
* **Changement d'heure** : l'index est en heure locale murale à 24 pas par jour ; au printemps l'heure inexistante est comblée par RTE (02:00 = 03:00), à l'automne l'heure répétée est absente ; ~0,5 % des jours, effet non mesuré. Les lags 168 h restent alignés sur l'heure d'horloge (tests dédiés).
* Le dashboard est validé en headless (aucune exception sur toutes les interactions testées, y compris l'onglet des paramètres) mais n'a pas été contrôlé visuellement dans un navigateur.

## 6. Prochaines étapes (par ordre de valeur)

1. **Vraies prévisions météo archivées** (Open-Meteo historical forecast, AROME/ARPEGE, ECMWF) pour remplacer `normal`/`noisy` : c'est le facteur qui change le plus les résultats, y compris la conclusion sur les règles. À défaut, un scénario « persistance de l'anomalie » (normale + anomalie récente) pour les features du ML et pour la normalisation des règles, et corriger le mode adaptatif pour qu'il conserve l'anomalie récente.
2. Confronter l'heure d'émission retenue (10:00) au calendrier réel de publication RTE.
3. **Allonger l'historique** : eCO2mix définitif 2012-2019 (+ Enedis/ERA5 pour la température) → 8 années de plus pour trouver de meilleurs analogues et estimer les coefficients de fériés (aujourd'hui 24 fériés).
4. **Décision go/no-go sur les règles** : les conserver seulement sur les fériés isolés (gain démontré contre le RF), puis re-tester S51–S2 et l'été avec des analogues moyennés (top-k pondérés par le score) et la météo prolongée plutôt qu'un seul analogue.
5. Élargir la grille d'hyperparamètres, ajouter des intervalles de prédiction (quantiles), essayer LightGBM/quantile regression.
6. Seulement ensuite : Airflow, Docker, MLflow, CI/CD, monitoring (Prometheus/Grafana) — et un modèle plus complexe **uniquement si** son gain hors échantillon justifie sa complexité opérationnelle : à ce stade, XGBoost bat déjà nettement les baselines classiques.

## 7. Modèles sauvegardés et reproduction

Les modèles entraînés sont dans `data/models/<scénario>/` (ignoré par git, **3,7 Go** au total, 1,3 Go par scénario) : pour chaque fold `F<nn>_<modèle>` (entraîné avant le fold) et `final_<modèle>` (entraîné sur toutes les données 2020-2024). XGBoost : `.ubj` (9-10 Mo) ; Random Forest : `joblib` compressé (53-66 Mo pour le modèle final) ; SARIMAX : JSON de la spécification et des coefficients (~1 Ko ; les résultats statsmodels complets pèsent 124 Mo par fold à cause du filtre de Kalman). `data/models/manifest.json` liste les variables attendues (dans l'ordre), l'`issue_hour`, les hyperparamètres et les versions des bibliothèques. Un modèle rechargé redonne exactement les prédictions du benchmark (écart 0,000000 MW, vérifié). Les paramètres (hyperparamètres, coefficients SARIMAX, année analogue et sensibilité thermique des règles) sont consultables dans l'onglet « Paramètres des modèles » du dashboard (`data/results/model_params.parquet`, `rule_params.parquet`).

```bash
python scripts/02_analyze_hypotheses.py
OMP_NUM_THREADS=1 nice -n 19 python scripts/03_run_benchmark.py --jobs 1 --save-models   # ~72 min sur 1 cœur
python scripts/04_bootstrap_compare.py --a random_forest_rules_adaptive --b random_forest
streamlit run app/dashboard.py
```
