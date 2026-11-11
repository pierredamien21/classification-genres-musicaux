# CNN de référence — entraînement et comparaison

23,816 paramètres, apprentissage de zéro, graine 42. 10 époques réalisées ; époque **9** retenue sur la macro-F1 par morceau. Même sélection filtrée : 6 299 morceaux de train et 780 de validation. Le test n’est pas évalué.

Motif d’arrêt enregistré : `max_epochs`. Statut : `complete`.

## Comparaison par morceau

| Système | Macro-F1 validation | Accuracy validation | Macro-F1 train |
|---|---:|---:|---:|
| Descripteurs | 0.4612 | 0.4756 | 0.5446 |
| ViT gelé | 0.5447 | 0.5538 | 0.6051 |
| Fusion complète | 0.5219 | 0.5295 | 0.6678 |
| Fusion sans rythme | 0.5387 | 0.5487 | 0.6577 |
| Fusion sans harmonie | 0.5282 | 0.5385 | 0.6582 |
| CNN de zéro | 0.5301 | 0.5577 | 0.5425 |

Écart CNN − ViT : **-0.0146** en macro-F1 et **+0.0038** en accuracy.
Le classement principal repose sur la macro-F1, fixée au plan : chaque genre y a le même poids. Une meilleure accuracy n’implique pas une meilleure macro-F1, car les erreurs peuvent être réparties différemment entre genres.

## Historique complet

| Époque | Loss train par fenêtre | Macro-F1 validation | Accuracy validation | Durée (s) |
|---:|---:|---:|---:|---:|
| 1 | 1.8472 | 0.3165 | 0.3603 | 79.7 |
| 2 | 1.6544 | 0.3978 | 0.4295 | 74.9 |
| 3 | 1.5495 | 0.4370 | 0.4615 | 68.7 |
| 4 | 1.4938 | 0.4637 | 0.4897 | 67.1 |
| 5 | 1.4529 | 0.4425 | 0.4654 | 68.4 |
| 6 | 1.4259 | 0.4954 | 0.5218 | 68.0 |
| 7 | 1.3945 | 0.5132 | 0.5321 | 66.0 |
| 8 | 1.3709 | 0.5113 | 0.5321 | 67.7 |
| 9 | 1.3449 | 0.5301 | 0.5577 | 69.6 |
| 10 | 1.3221 | 0.5270 | 0.5397 | 65.5 |

La loss est mesurée pendant les mises à jour sur les fenêtres de train ; les scores de validation sont calculés ensuite, par morceau, en mode évaluation. Le score train du tableau principal est recalculé avec les poids de l’époque retenue.

## Résultats par genre

| Genre | Effectif | F1 ViT | F1 CNN | Rappel CNN |
|---|---:|---:|---:|---:|
| Electronic | 99 | 0.5536 | 0.5169 | 0.4646 |
| Experimental | 100 | 0.4571 | 0.5899 | 0.6400 |
| Folk | 99 | 0.6702 | 0.6667 | 0.7879 |
| Hip-Hop | 100 | 0.7343 | 0.6607 | 0.7400 |
| Instrumental | 93 | 0.5028 | 0.4371 | 0.3548 |
| International | 98 | 0.6598 | 0.6066 | 0.6531 |
| Pop | 94 | 0.1098 | 0.0870 | 0.0638 |
| Rock | 97 | 0.6701 | 0.6763 | 0.7216 |

## Confusion CNN

Lignes : genre réel ; colonnes : genre prédit.

| Réel / prédit | Electronic | Experimental | Folk | Hip-Hop | Instrumental | International | Pop | Rock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Electronic | 46 | 9 | 2 | 18 | 8 | 8 | 4 | 4 |
| Experimental | 3 | 64 | 6 | 2 | 6 | 6 | 1 | 12 |
| Folk | 0 | 2 | 78 | 2 | 3 | 5 | 7 | 2 |
| Hip-Hop | 11 | 0 | 0 | 74 | 0 | 10 | 3 | 2 |
| Instrumental | 1 | 14 | 22 | 1 | 33 | 3 | 14 | 5 |
| International | 6 | 3 | 8 | 11 | 3 | 64 | 2 | 1 |
| Pop | 12 | 19 | 14 | 14 | 4 | 11 | 6 | 14 |
| Rock | 0 | 6 | 5 | 2 | 1 | 6 | 7 | 70 |

CNN correct et ViT incorrect : 93 morceaux ; ViT correct et CNN incorrect : 90.

## Coût et vérifications

- Extraction du cache log-Mel : 3.72 min.
- Entraînement et validation : 11.60 min.
- Avec export et évaluation finale du train : 12.07 min.
- Écart maximal entre probabilités avant/après export : 0.

Rapports, matrice log-Mel, index, poids et prédictions vérifiés par empreintes. Même index de fenêtres pour toutes les références. Métriques recalculées depuis les probabilités exportées ; meilleure époque revérifiée sur tout l’historique.

## Limites et suite

Ce CNN est une référence de développement à une seule graine, limitée à dix époques. Le motif max_epochs ne démontre pas une convergence ou une performance maximale. Le benchmark CPU a été respecté sans ouvrir le jeu de test.

CNN : log-Mel natif, apprentissage de zéro, sélection d’époque. ViT : log-Mel redimensionné, poids image préentraînés et gelés, sélection de C. Les budgets de sélection diffèrent ; c’est une comparaison de systèmes, pas une ablation de l’architecture seule. La validation sert au développement ; aucun écart n’est présenté comme statistiquement établi.

Prochain jalon : ablation ViT fenêtre centrale seule contre trois fenêtres, en réutilisant les embeddings existants, avec les mêmes morceaux et la même grille C. Les autres ablations, répétitions stochastiques, verrouillage du test et démo restent à réaliser.

## Reproduction et preuves

Voir [le protocole CNN](../docs/archive/CNN.md).

- Rapport CNN complet (`runs/cnn-20261020T202753531958Z/report.json`, non versionné)
- Modèle exporté (`runs/cnn-20261020T202753531958Z/model.safetensors`, non versionné)
- [Comparaison vérifiable](CNN_COMPARISON.json)
