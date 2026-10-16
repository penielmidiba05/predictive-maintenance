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
- Semaine 1 (à partir du 19/10) : labels (RUL, cible à 30 cycles), split par moteur, baseline