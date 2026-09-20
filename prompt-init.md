Crée un projet Python `rte-load-forecasting-demo`.

OBJECTIF
Construire un démonstrateur end-to-end de prévision de la consommation électrique française, orienté production/MLOps.

Pendant mon absence, avance de manière autonome jusqu'à obtenir un premier benchmark fonctionnel.

1. DATA
- ouvre tous les fichiers eco2mix qui contiennent les consommations de RTE eCO2mix_RTE_Annuel-Definitif_20XX.zip
- Identifie une source de température historique ici sur le site d'enedis via leur API  :https://opendata.enedis.fr/datasets/donnees-de-temperature-et-de-pseudo-rayonnement/api-doc.
- et, séparément, de prévisions météo ce n'est pas le point critique si tu ne trouves pas une source exploitable rapidment on commencera par dire que la préviions de température est égale à la température normale du jeu de données enedis 
- Stocke les données brutes et préparées en Parquet.
- Documente précisément les sources.

2. FEATURES
Construis sans data leakage :
- hour
- weekday
- month
- holidays France
- load lag 24h
- load lag 48h
- load lag même jour de semaine la semaine dernière
- load lag même jour de semaine la semaine S-2
- rolling means
- température normale et réalisée 
- température noramle² et réalisée²
- interactions température/calendrier

3. BASELINES
Implémente :
A. persistence J-7
B. SARIMAX avec température comme variable exogène
C. XGBoost 
D. XGBoost + règles de métier

r# OBJECTIF

Construire un modèle de prévision de la consommation électrique française dans lequel :

1. le **Random Forest** apprend le comportement normal de la consommation ;
2. les périodes calendaires très atypiques sont traitées par un **moteur de règles métier explicites** ;
3. les jours fériés et jours de pont ne contaminent pas l'apprentissage du comportement d'un jour normal ;
4. toutes les règles restent explicables, testables et backtestables.

L'objectif n'est donc PAS de demander au Random Forest d'apprendre seul Noël, le Nouvel An, les ponts ou la baisse estivale d'août à partir de quelques observations historiques.

La logique cible est :

```text
Données historiques
        ↓
Nettoyage / calendrier
        ↓
Identification des périodes spéciales
        ↓
┌───────────────────────────────┐
│ Journée/période normale ?     │
└──────────────┬────────────────┘
               │
        oui    │    non
               │
               ↓
      Random Forest      Moteur règles métier
               │              │
               └──────┬───────┘
                      ↓
                Forecast final
```

# 1. CLASSIFICATION CALENDAIRE

Pour chaque timestamp, créer au minimum :

```python
year
month
iso_week
weekday
hour
is_weekend
is_public_holiday
is_bridge_day
is_ramp_up
is_ramp_down
is_august_special_period
special_period_type
```

Définition d'un **jour de pont** :

> Jour ouvré situé directement entre un jour férié et un week-end, ou entre un week-end et un jour férié.

Exemples :

```text
jeudi férié + vendredi ouvré + samedi
→ vendredi = pont

dimanche + lundi ouvré + mardi férié
→ lundi = pont
```

Toutes ces classifications doivent être déterminées **avant l'entraînement**.

---

# 2. RÈGLE FONDAMENTALE D'APPRENTISSAGE

## Journées normales

Le Random Forest apprend le comportement des journées normales.

## Jours fériés et ponts

Les observations correspondant à :

```text
is_public_holiday == True
OR
is_bridge_day == True
```

doivent être **exclues de l'échantillon utilisé pour apprendre le comportement d'une journée normale**.

Autrement dit, si je cherche à prévoir un mardi normal, un mardi férié historique ne doit PAS être considéré comme un mardi comparable.

Même principe pour un jour de pont.

IMPORTANT :

Ne pas simplement créer :

```python
is_holiday = 1
```

et laisser le Random Forest tout apprendre.

Je veux explicitement séparer :

```text
apprentissage comportement normal
```

et :

```text
traitement des événements calendaires exceptionnels
```

---

# 3. RAMP-DOWN DE FIN D'ANNÉE : S51 ET S52

Les semaines ISO 51 et 52 doivent être traitées comme une période spéciale.

Le Random Forest **passe la main au moteur de règles métier**.

## Étape 1 : trouver une année historique analogue

Pour l'année cible Y, rechercher dans l'historique une année Y_ref dont la configuration calendaire de Noël et du Nouvel An est la plus proche.

Exemple cible :

```text
2026
Noël : vendredi
Nouvel An : vendredi
```

Une année historique où ces jours tombent jeudi ou vendredi est beaucoup plus pertinente qu'une année où ils tombent lundi.

Créer une fonction :

```python
find_best_year_end_calendar_analogue(target_year)
```

Le score de similarité doit au minimum considérer :

```text
weekday(Christmas)
weekday(New Year)
position dans les semaines ISO
structure week-end / férié / pont
```

La méthode et le score doivent être documentés.

---

# 4. ESTIMATION DU NIVEAU DE S51/S52

Avant de déterminer la forme horaire, estimer le **niveau général de consommation** attendu en S51 et S52 de l'année cible.

Utiliser comme référence les semaines précédentes :

```text
S48
S49
S50
```

Mais AVANT de calculer le niveau de référence, nettoyer ces semaines.

## Rebouchage des données

Si S48/S49/S50 contiennent :

* données manquantes ;
* trous ;
* observations manifestement invalides ;

les reconstruire avant de calculer le niveau de référence.

La méthode d'imputation doit être explicite et ne doit jamais utiliser d'information future qui n'aurait pas été disponible au moment du forecast.

Produire ensuite :

```python
baseline_pre_christmas
target_level_S51
target_level_S52
```

Le niveau des semaines S51/S52 doit être estimé relativement au niveau normal précédant la période de Noël.

---

# 5. TRANSFERT DE FORME DEPUIS L'ANNÉE ANALOGUE

Une fois le niveau cible de S51/S52 déterminé, utiliser l'année analogue pour transférer **la forme relative**, pas son niveau absolu.

Exemple :

Si dans l'année analogue :

```text
mean_load_monday_S51_ref
------------------------- = 0.90
mean_load_S51_ref
```

alors :

```text
mean_load_monday_S51_target
=
0.90 × target_level_S51
```

Même logique pour chaque jour.

---

# 6. TRANSFERT DE LA FORME HORAIRE

Descendre ensuite au niveau horaire.

Pour chaque journée de l'année analogue, calculer un profil normalisé :

```text
load(hour)
----------------
mean_load(day)
```

Exemple :

```text
lundi S51 analogue

00h → 0.72
01h → 0.68
...
09h → 1.12
...
19h → 1.18
...
```

Appliquer cette forme au niveau journalier cible.

On obtient donc une construction hiérarchique :

```text
niveau avant vacances
        ↓
niveau semaine
        ↓
coefficient journal / semaine
        ↓
profil horaire normalisé
        ↓
forecast horaire
```

Cette décomposition est importante et doit rester visible dans le code.

---

# 7. TRAITEMENT DE NOËL ET DU NOUVEL AN

Même principe.

Si, dans l'année analogue :

```text
mean_load_christmas_ref
-----------------------
mean_load_S51_ref
= 0.80
```

alors :

```text
mean_load_christmas_target
=
0.80 × target_level_S51
```

Puis transférer le **profil intraday normalisé du jour férié analogue**.

IMPORTANT :

Le jour férié analogue doit être choisi en tenant compte de sa position dans la semaine.

Un Noël vendredi doit prioritairement être comparé à un Noël jeudi/vendredi plutôt qu'à un Noël lundi.

---

# 8. RAMP-UP : S1 ET S2

Appliquer une logique symétrique au redémarrage de janvier.

S1/S2 ne doivent pas être traitées comme des semaines ordinaires par le Random Forest.

Identifier une année historique analogue selon :

```text
weekday(New Year)
structure week-end / jour férié
jours de pont éventuels
```

Puis reconstruire progressivement :

```text
niveau Nouvel An
        ↓
niveau S1
        ↓
niveau S2
        ↓
retour vers niveau normal
```

La forme historique analogue doit être transférée sous forme de coefficients relatifs et non en MW absolus.

---

# 9. PÉRIODE ESTIVALE / AOÛT

Le mois d'août possède une dynamique particulière.

## Identification du point bas

Pour chaque année historique, identifier la semaine dont le week-end est **le plus proche du 15 août**.

Cette semaine constitue le candidat principal au :

```text
summer_trough_week
```

et doit correspondre au voisinage du point bas annuel de consommation.

Vérifier empiriquement cette hypothèse sur les données historiques et signaler les années où elle n'est pas vérifiée.

Ne jamais masquer une exception historique pour forcer la règle.

---

# 10. CONSTRUCTION DE LA RAMPE ESTIVALE

Le forecast d'été doit distinguer :

```text
niveau normal avant vacances
        ↓
RAMP-DOWN été
        ↓
point bas autour du 15 août
        ↓
RAMP-UP été
        ↓
retour au niveau normal
```

Déterminer :

```python
summer_start_level
summer_trough_level
summer_end_level
```

Le niveau de départ doit être calibré à partir des semaines précédant réellement le départ en vacances.

Le niveau du point bas doit être estimé à partir des années historiques comparables.

Les semaines intermédiaires doivent être construites relativement à ces niveaux.

Éviter une simple interpolation arbitraire si les données historiques permettent d'estimer une forme moyenne de ramp-down/ramp-up.

---

# 11. AUTRES JOURS FÉRIÉS

Pour les jours fériés hors périodes spéciales :

```text
Noël / Nouvel An
Ramp-up S1/S2
Ramp-down S51/S52
période estivale
```

ne pas laisser automatiquement le Random Forest extrapoler comme s'il s'agissait d'une journée normale.

Créer un module séparé :

```python
forecast_isolated_holiday(...)
```

Le principe devra être :

1. déterminer le niveau attendu autour du jour férié à partir du forecast normal ;
2. rechercher des occurrences historiques comparables du même férié ;
3. privilégier les occurrences tombant sur un weekday identique ou proche ;
4. mesurer le ratio :

```text
load holiday
------------
normal surrounding level
```

5. appliquer le coefficient historique/robuste au niveau cible ;
6. reconstruire le profil horaire à partir de profils historiques normalisés.

IMPORTANT :

Ne pas inventer maintenant une règle arbitraire si les données ne permettent pas de la justifier.

Produire les statistiques historiques et proposer la règle à partir des observations.

---

# 12. RANDOM FOREST

Le Random Forest doit apprendre la relation :

```text
normal calendar
+ weather
+ autoregressive load features
→ load
```

Features candidates :

```python
hour
weekday
month
week_of_year

temperature_forecast
temperature_forecast_squared

load_lag_24h
load_lag_48h
load_lag_168h

rolling_mean_24h
rolling_mean_7d
```

Attention absolue au **data leakage**.

Une feature utilisée pour prévoir t doit avoir été disponible au moment où la prévision de t aurait réellement été réalisée.

---

# 13. SPLIT ET BACKTEST

INTERDICTION d'utiliser :

```python
train_test_split(..., shuffle=True)
```

Utiliser un split strictement temporel.

Puis mettre en place un **walk-forward / rolling backtest**.

Mais surtout, évaluer séparément :

```text
NORMAL DAYS
HOLIDAYS
BRIDGE DAYS
SUMMER
S51/S52
S1/S2
```

Je veux pouvoir déterminer si le moteur hybride :

```text
Random Forest + business rules
```

est réellement meilleur qu'un Random Forest seul.

---

# 14. BENCHMARK OBLIGATOIRE

Comparer au minimum :

```text
Model A
naive J-7

Model B
Random Forest seul

Model C
Random Forest + règles métier
```

Produire :

```text
MAE
RMSE
bias
```

globalement ET par type de période.

Créer notamment un tableau :

| Period      | Naive J-7 | RF | RF + Business Rules |
| ----------- | --------: | -: | ------------------: |
| Normal days |           |    |                     |
| Holidays    |           |    |                     |
| Bridge days |           |    |                     |
| August      |           |    |                     |
| S51-S52     |           |    |                     |
| S1-S2       |           |    |                     |
| ALL         |           |    |                     |

---

# 15. ARCHITECTURE CODE

Ne pas mélanger les règles métier dans le code du Random Forest.

Je veux quelque chose proche de :

```text
src/
├── calendar/
│   ├── holidays.py
│   ├── bridge_days.py
│   └── special_periods.py
│
├── data/
│   ├── cleaning.py
│   └── imputation.py
│
├── features/
│   └── build_features.py
│
├── models/
│   ├── random_forest.py
│   └── baseline.py
│
├── business_rules/
│   ├── year_end.py
│   ├── year_start.py
│   ├── summer.py
│   └── isolated_holidays.py
│
├── evaluation/
│   ├── backtest.py
│   └── metrics.py
│
└── forecast.py
```

`forecast.py` doit agir comme routeur :

```python
if is_year_end_period:
    forecast_year_end()

elif is_year_start_period:
    forecast_year_start()

elif is_summer_special_period:
    forecast_summer()

elif is_public_holiday or is_bridge_day:
    forecast_isolated_holiday()

else:
    random_forest.predict()
```

---

# 16. TESTS MÉTIER OBLIGATOIRES

Créer des tests automatisés vérifiant notamment :

```text
un jour férié n'entre pas dans le training set normal
un jour de pont n'entre pas dans le training set normal
aucune donnée future n'entre dans les features
S51/S52 déclenchent bien le moteur year_end
S1/S2 déclenchent bien le moteur year_start
la semaine proche du 15 août est correctement identifiée
le choix de l'année analogue est reproductible
les coefficients journaliers sont calculés relativement au bon niveau hebdomadaire
les profils horaires normalisés se recomposent correctement
```

---

# 17. EXPLICABILITÉ

Pour chaque forecast produit par une règle métier, je veux pouvoir afficher :

```text
Forecast date:
2026-12-25

Forecast method:
YEAR_END_BUSINESS_RULE

Reference year:
2025

Reason:
Christmas weekday proximity

Weekly target level:
52 300 MW

Holiday coefficient:
0.81

Hourly profile source:
Christmas 2025

Final forecast:
...
```

Le système ne doit donc jamais produire un chiffre spécial sans pouvoir expliquer **d'où il vient**.

---

# 18. PRIORITÉ D'IMPLÉMENTATION

Ne code pas tout aveuglément.

Procéder dans cet ordre :

1. analyser les données historiques ;
2. créer le calendrier et classifier les périodes ;
3. vérifier empiriquement les hypothèses métier décrites ci-dessus ;
4. afficher les statistiques et anomalies trouvées ;
5. implémenter la baseline Random Forest ;
6. implémenter S51/S52 ;
7. implémenter S1/S2 ;
8. implémenter août ;
9. analyser puis implémenter les autres jours fériés ;
10. effectuer le backtest comparatif.

**Si une règle métier décrite ici est contredite par les données historiques, ne la corrige pas silencieusement. Signale-le explicitement et montre les données qui posent problème.**

## Résultat recherché

Je veux pouvoir expliquer le système ainsi :

> **Le Random Forest modélise le régime normal de consommation. Je ne lui demande pas d'apprendre correctement des événements qui n'arrivent qu'une fois par an à partir de quelques observations. Les régimes calendaires rares sont identifiés séparément et reconstruits à partir d'années analogues, en séparant niveau hebdomadaire, effet journalier et forme intraday. Ensuite je backteste le modèle hybride contre le Random Forest seul pour vérifier que cette connaissance métier apporte effectivement de la valeur hors échantillon.**


4. VALIDATION
INTERDICTION de random train/test split.

Implémente :
train → validation → test chronologiques

puis une rolling / expanding-window cross-validation.

Calcule :
- MAE
- RMSE
- bias
- training time

5. VISUALISATIONS
Génère des graphiques très lisibles :
- train / validation / test
- actual vs J-7 vs SARIMAX vs XGBoost
- erreurs
- MAE par heure
- performance de chaque fold CV
- training time
- impact des hyperparamètres
- feature importance

Si SHAP est facilement disponible, ajoute :
- SHAP summary
- SHAP dependence plot température

6. ARCHITECTURE

src/
  data/
  features/
  models/
  evaluation/

tests/
configs/
notebooks/
data/
.github/workflows/

pyproject.toml
README.md

7. TESTS
Ajoute notamment des tests contre :
- data leakage
- timestamps dupliqués
- NaN target
- incohérence des lags
- train postérieur au test

8. CI
Ajoute GitHub Actions :
ruff → pytest.

9. RAPPORT
Crée `RESULTS.md` avec :
- architecture
- données utilisées
- performances J-7 / SARIMAX / XGBoost
- temps d'entraînement
- principaux enseignements
- limites
- prochaines étapes

IMPORTANT :
Ne déploie PAS encore Airflow, FastAPI, Docker, MLflow,
Prometheus ou Grafana.

Je veux d'abord un pipeline ML simple, correct,
reproductible et testé.

Ne complexifie pas artificiellement le modèle.
Pendant que Claude bosse

Toi → déjeuner. 🍜

À ton retour, je ferais :

1. 30 min → audit de ce qu'il a produit et surtout recherche de data leakage.

2. 1 h → rendre le résultat ML très visuel et comprendre pourquoi XGBoost bat ou non SARIMAX.

3. Ensuite seulement → Airflow + Docker + GitHub Actions/CD.

Et surtout, ne lance pas PyTorch maintenant. Si XGBoost obtient déjà un excellent résultat, c'est justement une discussion intéressante avec Ambos :

« Je pars volontairement du modèle le plus simple capable de battre sérieusement la baseline. Je n'ajoute un modèle deep learning que si son gain out-of-sample justifie sa complexité opérationnelle. »

C'est beaucoup plus crédible que de coller un Transformer sur une série temporelle parce que l'offre contient « Deep Learning ».
