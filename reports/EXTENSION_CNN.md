# CNN : comparaison des plafonds de dix et vingt époques

Prolongation adaptative des trois runs, sans réinitialiser optimiseur, patience ou générateur aléatoire.

| Graine | Époques réalisées | Retenue avant → après | F1 avant | F1 après | Différence | Arrêt |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 18 | 9 → 15 | 0.5301 | 0.5464 | +0.0163 | early_stopping |
| 43 | 13 | 10 → 10 | 0.5471 | 0.5471 | +0.0000 | early_stopping |
| 44 | 11 | 8 → 8 | 0.5287 | 0.5287 | +0.0000 | early_stopping |

## Moyenne et dispersion

- macro_f1 : **0.5407 ± 0.0105** ; différence moyenne appariée : +0.0054.
- accuracy : **0.5560 ± 0.0075** ; différence moyenne appariée : +0.0013.

Écart-type d’échantillon entre trois graines ; aucun ensemble ni choix de la meilleure graine.

## Genres

| Genre | F1 moyenne avant | Après | Écart-type après | Différence |
|---|---:|---:|---:|---:|
| Electronic | 0.5138 | 0.5065 | 0.0191 | -0.0073 |
| Experimental | 0.5789 | 0.5728 | 0.0412 | -0.0061 |
| Folk | 0.6877 | 0.6836 | 0.0262 | -0.0041 |
| Hip-Hop | 0.6566 | 0.6705 | 0.0291 | +0.0139 |
| Instrumental | 0.4268 | 0.4478 | 0.0512 | +0.0210 |
| International | 0.6299 | 0.6410 | 0.0074 | +0.0111 |
| Pop | 0.0909 | 0.0959 | 0.0254 | +0.0050 |
| Rock | 0.6979 | 0.7080 | 0.0057 | +0.0100 |

## Coût observé

| Graine | Apprentissage + validation supplémentaires (min) | Avec export final et évaluation train (min) |
|---:|---:|---:|
| 42 | 9.19 | 9.65 |
| 43 | 3.45 | 3.92 |
| 44 | 1.26 | 1.73 |

Temps cumulés des invocations d’apprentissage après le checkpoint parent, hors initialisation et vérification initiale du cache. Les pauses entre invocations ne sont pas comptées. L’estimation initiale avec marge était de 69 minutes ; les arrêts anticipés peuvent la réduire.

## Courbes

*Figure non versionnée : `reports/figures/cnn_extension.png`*

Figure PDF (`reports/figures/cnn_extension.pdf`, non versionné). Les dix premières époques sont héritées et vérifiées à l’identique.

## Interprétation et limites

L’extension a été décidée après examen des courbes à dix époques. Elle est une expérience de développement adaptative, pas une confirmation indépendante. Le meilleur score est recherché parmi davantage d’époques : une hausse sur cette validation ne démontre pas à elle seule une meilleure généralisation.

Le plafond est vingt époques ; la patience reste trois. Un arrêt anticipé ne prouve pas la convergence maximale de l’architecture. Trois graines ne permettent pas de revendiquer une significativité statistique sur les artistes. Le test reste non chargé et non évalué.

## Vérifications et reproduction

Parents, checkpoints, configurations, sources, poids et prédictions contrôlés par empreintes. Métriques recalculées ; historique initial, règle d’arrêt, époque choisie et rechargement des exports vérifiés.

Voir [le protocole](../docs/archive/EXTENSION_CNN.md), [la référence à dix époques](REPETITIONS_CNN.md) et [la synthèse globale](SYNTHESE_VALIDATION.md).

Commande : `.\.venv\Scripts\python.exe scripts/summarize_cnn_extension.py`.

- [Résumé numérique](CNN_EXTENSION.json)
- Rapport graine 42 (`runs/cnn20_seed42-20261021T185433222368Z/report.json`, non versionné)
- Rapport graine 43 (`runs/cnn20_seed43-20261021T190439492087Z/report.json`, non versionné)
- Rapport graine 44 (`runs/cnn20_seed44-20261021T190901243607Z/report.json`, non versionné)
