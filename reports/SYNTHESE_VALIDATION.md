# Synthèse des expériences de développement

Meilleure macro-F1 de validation actuelle : **ViT bloc 6, canaux HPSS**, **0.5975**. Ce classement sert au développement ; le jeu de test reste non évalué.

## Résultats sur les mêmes morceaux

Toutes les expériences utilisent 6 299 morceaux de train et 780 de validation. La fenêtre centrale utilise un extrait par morceau ; les autres systèmes en utilisent trois. Les normalisations apprises sont ajustées uniquement sur train.

| Système | Macro-F1 validation | Accuracy validation | Macro-F1 train | Sélection |
|---|---:|---:|---:|---|
| ViT bloc 6, canaux HPSS | 0.5975 | 0.6128 | 0.6352 | C = 0.1 |
| ViT bloc 12, canaux HPSS | 0.5844 | 0.5974 | 0.6042 | C = 1 |
| ViT bloc 6, trois fenêtres | 0.5780 | 0.5885 | 0.6394 | C = 0.1 |
| ViT bloc 12, trois fenêtres | 0.5447 | 0.5538 | 0.6051 | C = 0.1 |
| CNN, plafond 20, moyenne de trois graines | 0.5407 | 0.5560 | 0.5623 | graines 42, 43, 44 |
| Fusion sans rythme | 0.5387 | 0.5487 | 0.6577 | C = 1 |
| CNN, moyenne de trois graines | 0.5353 | 0.5547 | 0.5432 | graines 42, 43, 44 |
| Fusion sans harmonie | 0.5282 | 0.5385 | 0.6582 | C = 0.1 |
| Fusion bloc 12 + descripteurs | 0.5219 | 0.5295 | 0.6678 | C = 0.1 |
| ViT bloc 12, fenêtre centrale | 0.4882 | 0.4962 | 0.5748 | C = 0.1 |
| Descripteurs musicaux | 0.4612 | 0.4756 | 0.5446 | C = 0.1 |
| Prédicteur majoritaire | 0.0284 | 0.1282 | — | classe de train |

CNN : macro-F1 moyenne **0.5353 ± 0.0103** (écart-type d’échantillon entre trois graines). Le tableau utilise la moyenne des métriques des trois réseaux, pas un ensemble ni la meilleure graine. Voir [les répétitions](REPETITIONS_CNN.md) pour chaque run et les limites.

CNN, plafond vingt époques : **0.5407 ± 0.0105**, différence moyenne appariée +0.0054. Cette prolongation a été décidée après inspection des courbes initiales ; elle augmente le nombre d’époques candidates sur validation. Voir [l’extension CNN](EXTENSION_CNN.md).

## Ce que les comparaisons permettent de conclure

- Fusion : l’apport des descripteurs est étudié avec le bloc 12, ainsi que le retrait du rythme ou de l’harmonie. Aucune fusion avec le bloc 6 n’a été testée.
- Fenêtres : centrale seule contre trois fenêtres au bloc 12 ; cela modifie aussi le nombre d’exemples d’apprentissage, pas seulement l’inférence.
- Profondeur : bloc 6 contre bloc 12 à canaux identiques, avec la convention de LayerNorm finale gelée documentée.
- Canaux : original/H/P contre original répété, aux blocs 12 et 6. La combinaison bloc 6 + HPSS a été décidée après les ablations initiales ; elle complète les quatre configurations profondeur × canaux, avec C choisi séparément dans la même grille. Les gains ne sont pas supposés additifs.
- CNN contre ViT : architectures, préentraînements et modes d’apprentissage différents. Cette comparaison ne prouve pas un effet isolé de l’architecture.

Les métriques de validation ont été recalculées depuis les probabilités sauvegardées ; rapports, modèles et prédictions sont vérifiés par empreintes. Les mêmes identifiants et étiquettes de validation sont exigés. Les comparateurs propres à chaque ablation contrôlent en plus leurs caches et leurs paramètres.

## Limites et prochaines étapes

C et les époques ont été sélectionnés sur validation. Les résultats du tableau sont donc des estimations de développement, sans intervalle de confiance final ni affirmation de significativité. Répéter les régressions déterministes ne constituerait pas une mesure de variabilité comparable.

Le budget initial des deux extractions supplémentaires d’ablation est consommé. Le plan reste évolutif : un point de décision sur les erreurs et les nouvelles hypothèses précède le verrouillage. Les extensions retenues auront un protocole et un budget explicites. Viendront ensuite les analyses de robustesse et l’évaluation du test avec bootstrap groupé par artiste. La démo utilisera le même prétraitement et le même artefact exporté.

## Rapports et reproduction

Commande : `.\.venv\Scripts\python.exe scripts/summarize_validation.py`.

- [Résumé machine et empreintes des expériences](VALIDATION_SUMMARY.json)
- [ViT et descripteurs](COMPARAISON_VIT.md)
- [Fusion](COMPARAISON_FUSION.md)
- [CNN](BASELINE_CNN.md)
- [Fenêtres](ABLATION_FENETRES.md)
- [Blocs](ABLATION_BLOCS.md)
- [Canaux HPSS](ABLATION_HPSS.md)
- [Combinaison bloc 6 + HPSS](COMBINAISON_BLOC6_HPSS.md)
- [Corrections et régressions de la combinaison](REGRESSIONS_BLOC6_HPSS.md)
