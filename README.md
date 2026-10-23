# predictive-maintenance
Prédiction de panne de moteurs à partir de capteurs (NASA C-MAPSS)
## Résultats

Validation croisée groupée (5 plis, 80 moteurs) — features glissantes (fenêtres 5, 10, 20), 112 features.

| Modèle | PR-AUC (CV) | Seuil | Rappel | Précision |
|---|---|---|---|---|
| Baseline (semaine 1, capteurs bruts) | [0.95] | 0,50 | [0.841] | [0.876] |
| Régression logistique | [0.969] ± [0.009] | [0.19] | [0.977] | [0.748] |
| Random Forest | [0.954] ± [0.013] | [0.23] | [0.980] | [0.661] |
| LightGBM | [0.961] ± [0.011] | [0.01] | [0.957] | [0.753] |
| LightGBM optimisé (Optuna, 30 essais) | [0.964] ± [-] | [0.01] | [-] | [-] |

**Évaluation finale (20 moteurs jamais vus, mesurée une seule fois)** — LightGBM optimisé :
PR-AUC [0.969], rappel [0.975], précision [0.750], seuil [0.01].

Le seuil minimise un coût où une panne ratée vaut 10 fois une maintenance inutile (hypothèse).

**Limites** : 20 moteurs de test seulement ; données simulées (FD001, une seule condition de vol) ; coût compté par cycle et non par moteur.