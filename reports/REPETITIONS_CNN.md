# CNN : stabilité sur trois graines

Même protocole, mêmes morceaux ; seules initialisation et permutation des lots changent.

| Graine | Époques réalisées | Époque choisie | Macro-F1 validation | Accuracy | Macro-F1 train |
|---:|---:|---:|---:|---:|---:|
| 42 | 10 | 9 | 0.5301 | 0.5577 | 0.5425 |
| 43 | 10 | 10 | 0.5471 | 0.5590 | 0.5539 |
| 44 | 10 | 8 | 0.5287 | 0.5474 | 0.5331 |

## Moyenne et dispersion entre graines

- macro_f1 : **0.5353 ± 0.0103** ; minimum 0.5287, maximum 0.5471.
- accuracy : **0.5547 ± 0.0063** ; minimum 0.5474, maximum 0.5590.

Écart-type d’échantillon, n − 1 = 2. Ces valeurs résument trois entraînements, pas un ensemble de réseaux ni un intervalle de confiance sur les artistes. L’époque de chaque run est choisie sur validation ; aucune graine n’est éliminée.

## F1 par genre

| Genre | Graine 42 | Graine 43 | Graine 44 | Moyenne | Écart-type |
|---|---:|---:|---:|---:|---:|
| Electronic | 0.5169 | 0.4961 | 0.5285 | 0.5138 | 0.0164 |
| Experimental | 0.5899 | 0.6147 | 0.5323 | 0.5789 | 0.0423 |
| Folk | 0.6667 | 0.7052 | 0.6911 | 0.6877 | 0.0195 |
| Hip-Hop | 0.6607 | 0.6455 | 0.6636 | 0.6566 | 0.0097 |
| Instrumental | 0.4371 | 0.4458 | 0.3976 | 0.4268 | 0.0257 |
| International | 0.6066 | 0.6489 | 0.6341 | 0.6299 | 0.0215 |
| Pop | 0.0870 | 0.1176 | 0.0680 | 0.0909 | 0.0250 |
| Rock | 0.6763 | 0.7032 | 0.7143 | 0.6979 | 0.0195 |

## Conséquence pour la suite

Pour 2 graine(s) sur trois, l’époque retenue appartient aux deux dernières époques du budget : [42, 43]. Cela signale une limite de durée à examiner, sans démontrer à lui seul un défaut de convergence.

Piste prioritaire issue de ces résultats : préparer une comparaison avec un plafond de vingt époques, sur les mêmes graines et sans changer architecture, optimiseur ou patience. Le coût doit être vérifié et le nouveau protocole consigné avant l’exécution. Cette extension n’est pas réalisée dans le présent rapport. Elle peut modifier le classement ; le test reste réservé.

## Courbes d’apprentissage

![Courbes des trois graines](figures/cnn_repetitions.png)

[Figure PDF](figures/cnn_repetitions.pdf). Les étoiles indiquent les époques sélectionnées.

## Contrôles et limites

Rapports, poids et prédictions vérifiés par empreintes ; métriques recalculées, sélection d’époque revérifiée, validation alignée avec le cache. Versions numériques, modèle, code d’apprentissage, données et configuration identiques sauf graine. Le script a seulement reçu une option de graine et des pointeurs séparés ; les empreintes des deux versions sont conservées dans le protocole.

La graine 42 provient du développement antérieur et est réutilisée, sans prétendre qu’il s’agit de trois nouveaux runs indépendants du choix de la recette. Le plafond de dix époques ne démontre pas la convergence maximale du CNN. Les durées dépendent de la charge du poste ; aucune comparaison contrôlée de vitesse n’est revendiquée.

Le test reste non évalué. Un point de décision sur validation précède tout verrouillage : analyse des erreurs, nouvelles hypothèses et budget peuvent faire évoluer le plan.

## Reproduction

Voir [le protocole](../docs/archive/REPETITIONS_CNN.md).

- [Rapport graine 42](../runs/cnn-20261020T202753531958Z/report.json)
- [Rapport graine 43](../runs/cnn_seed43-20261021T182143235893Z/report.json)
- [Rapport graine 44](../runs/cnn_seed44-20261021T183501483565Z/report.json)
- [Résumé numérique](CNN_REPETITIONS.json)
