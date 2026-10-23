# Journal

## Semaine 0 — 15-16 octobre 2026

**Fait**
- Dépôt GitHub créé (structure src/, tests/, notebooks/…)
- Environnement .venv + requirements.txt + requirements.lock.txt
- Données NASA C-MAPSS FD001 téléchargées dans data/raw (non commitées)
- Notebook 01_eda : chargement + exploration

**Bloqué**
- Activation du venv refusée par PowerShell → résolu avec Set-ExecutionPolicy RemoteSigned
- SHAP bloqué par le contrôle des applications Windows → reporté à la semaine 3 (WSL2 envisagé)

**Appris**
- 20 631 lignes mais seulement 100 moteurs → découper par moteur, pas par ligne
- Durées de vie de 128 à 362 cycles → l'âge seul ne suffit pas à prédire la panne
- Colonnes constantes à retirer : ... (ma liste)
- Les capteurs sont bruités mais ont une tendance en fin de vie → moyennes glissantes utiles

**Prochaine étape**

## Semaine 1 — 17-20 octobre 2026

**Fait**
- `src/data/load.py` : chargement des données C-MAPSS, liste des 14 capteurs utiles
- `src/data/labels.py` : calcul du RUL et de la cible (panne dans les 30 cycles)
- `src/data/split.py` : découpage par moteur, 80 moteurs de développement et 20 de test local (`configs/split_FD001.json`)
- 10 tests pytest (labels et découpage)
- Notebook `02_baseline` : régression logistique évaluée en validation croisée groupée (GroupKFold, 5 plis)

**Résultats**
- Taux de positifs : [0.151]
- PR-AUC : [0.950] avec `class_weight="balanced"`, [0.950] sans
- Rappel : [0.944] avec, [0.841] sans ; précision : [0.732] avec, [0.876] sans

**Bloqué**
- `load.py` créé dans `data/` au lieu de `src/data/` → `ModuleNotFoundError`. Corrigé en déplaçant le fichier. À retenir : `data/` est ignoré par Git, le code va toujours dans `src/`.
- `df.copy` sans parenthèses → `'function' object has no attribute 'groupby'`. À retenir : `objet.methode` désigne la méthode, `objet.methode()` l'exécute.
- `DATA.exists` sans parenthèses dans le `skipif` : bug silencieux, le test n'aurait jamais été ignoré sur la CI.
- SHAP toujours bloqué par le contrôle des applications Windows → à régler avant la semaine 3 (WSL2 envisagé).

**Appris**
- 20 631 lignes mais seulement 100 moteurs indépendants → découper par moteur pour éviter la fuite de données.
- `groupby().transform()` garde une valeur par ligne, contrairement à `agg()`.
- PR-AUC plutôt qu'accuracy sur des classes déséquilibrées : le hasard vaut le taux de positifs, pas 0,5.
- Le scaler doit être dans le Pipeline pour être ajusté uniquement sur les plis d'entraînement.
- `class_weight` déplace le compromis précision/rappel au seuil de 0,5, sans changer beaucoup la PR-AUC.
- Lire une erreur pytest : résumé en bas, ligne `>`, ligne `E`.


## Semaine 2 — 20-23 octobre 2026

**Fait**
- `src/features/build.py` : moyennes et écarts-types glissants (fenêtres 5, 10, 20) et différence au cycle précédent, par moteur → 112 features
- Briques communes : `configs/train.yaml`, `src/config.py`, `src/data/dataset.py`, `src/models/factory.py`, `src/evaluation/cv.py`, `src/evaluation/metrics.py`, `src/models/threshold.py`
- `src/models/train.py` : 3 modèles comparés en validation croisée groupée, suivis dans MLflow
- `src/models/tune.py` : tuning Optuna de LightGBM ([XX] essais), meilleurs paramètres dans `configs/lightgbm_best.yaml`
- `src/models/evaluate.py` : seuil choisi sur le dev, modèle final entraîné sur les 80 moteurs, évaluation unique sur les 20 moteurs de test
- 18 tests verts ; README mis à jour avec le tableau des résultats

**Résultats (validation croisée, 80 moteurs)**
- Baseline semaine 1 : PR-AUC [0.950]
- Régression logistique : PR-AUC [0.969] ± [0.009], seuil [0.19]
- Random Forest : PR-AUC [0.954] ± [0.013], seuil [0.23]
- LightGBM : PR-AUC [0.961] ± [0.011], seuil [0.01]
- LightGBM optimisé : PR-AUC [0.964] ± [-], seuil [0.01]
- **Test final (20 moteurs, mesuré une fois)** : PR-AUC [0.969], rappel [0.975], précision [0.750]

**Bloqué**
- `feature_columns()` renvoyait 168 colonnes au lieu de 112 : les 21 capteurs étaient utilisés au lieu des 14 utiles (erreur dans [load.py / build.py]). Le test de comptage l'a détecté.
- `ModuleNotFoundError: src.models.threshold` : erreur dans le nom ou l'emplacement du fichier.
- `.gitignore` : `data/` et `models/` sans `/` ignoraient aussi `src/data/` et `src/models/`, dont le code n'était pas sur GitHub. Corrigé avec `/data/` et `/models/`.
- SHAP toujours bloqué par le contrôle des applications Windows → à régler en semaine 3 (WSL2).

**Appris**
- Les features glissantes résument la tendance d'un capteur bruité ; elles doivent être calculées par moteur et uniquement sur le passé (test `test_no_future_leakage`).
- Une seule fonction de features pour l'entraînement et l'API, pour éviter le décalage entraînement/service.
- Random Forest = moyenne d'arbres indépendants ; boosting = arbres successifs qui corrigent les erreurs des précédents.
- Le seuil se choisit d'après le coût des erreurs : quand rater une panne coûte cher, le seuil baisse.
- MLflow garde paramètres, métriques et graphiques de chaque run ; Optuna explore les hyperparamètres automatiquement.
- Le gain principal vient de du tuning
- Le jeu de test ne sert qu'une fois, sinon il devient un jeu de réglage.
- Tester les hypothèses du code (nombre de colonnes) attrape des bugs que les tests de calcul ne voient pas.
- Dans un `.gitignore`, un motif sans `/` au début s'applique à tous les niveaux.

**Prochaine étape**
- Semaine 3 : installer WSL2 pour débloquer SHAP, tests supplémentaires (schéma des données, modèle), interprétabilité SHAP, logging, ruff et pre-commit, Dockerfile.