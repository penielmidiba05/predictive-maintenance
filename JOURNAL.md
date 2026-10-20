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

**Prochaine étape**
- Semaine 2 : features glissantes, Random Forest et LightGBM, MLflow, choix du seuil par coût, tuning Optuna, évaluation unique sur les 20 moteurs de test.