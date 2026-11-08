# ViT : fenêtre centrale seule contre trois fenêtres

Même ViT gelé, même prétraitement, mêmes 6 299 morceaux de train et 780 de validation. La variante centrale utilise uniquement [10,20] secondes, à l’entraînement et à l’évaluation. La référence utilise [0,10], [10,20] et [20,30] secondes. Aucun morceau n’est supprimé.

## Ablation principale prévue au plan

| Fenêtres train → évaluation | Fenêtres de train | C retenu | Macro-F1 validation | Accuracy | Macro-F1 train |
|---|---:|---:|---:|---:|---:|
| 3 → 3 | 18897 | 0.1 | 0.5447 | 0.5538 | 0.6051 |
| 1 centrale → 1 centrale | 6299 | 0.1 | 0.4882 | 0.4962 | 0.5748 |

Écart trois fenêtres − centrale : **+0.0566** en macro-F1.

## Grille complète

| C | Macro-F1 trois fenêtres | Macro-F1 centrale |
|---:|---:|---:|
| 0.1 | 0.5447 | 0.4882 |
| 1 | 0.5438 | 0.4819 |
| 10 | 0.5438 | 0.4775 |

## Contrôle complémentaire à modèle fixe

Après le résultat principal, le classifieur déjà retenu sur trois fenêtres est réévalué sur les seules fenêtres centrales. Son scaler, ses coefficients et C sont inchangés. Ce diagnostic ne constitue pas un nouveau réglage ni un modèle retenu pour le test.

| Modèle entraîné sur | Évaluation sur | Macro-F1 validation | Accuracy |
|---|---|---:|---:|
| Trois fenêtres | Trois fenêtres | 0.5447 | 0.5538 |
| Trois fenêtres | Centrale seule | 0.4896 | 0.4962 |

Cette paire conserve le modèle et change uniquement les fenêtres moyennées à l’évaluation. L’ablation principale change aussi le nombre d’exemples et la diversité temporelle pendant l’entraînement : son écart ne mesure donc pas seulement un effet de moyenne des probabilités.

## F1 par genre

| Genre | Effectif | Trois fenêtres | Centrale seule |
|---|---:|---:|---:|
| Electronic | 99 | 0.5536 | 0.4796 |
| Experimental | 100 | 0.4571 | 0.4386 |
| Folk | 99 | 0.6702 | 0.5979 |
| Hip-Hop | 100 | 0.7343 | 0.6404 |
| Instrumental | 93 | 0.5028 | 0.4556 |
| International | 98 | 0.6598 | 0.5918 |
| Pop | 94 | 0.1098 | 0.1013 |
| Rock | 97 | 0.6701 | 0.6000 |

Trois fenêtres correctes, centrale incorrecte : 86 morceaux ; centrale correcte, trois fenêtres incorrectes : 41.

## Calcul, vérifications et limites

Entraînement/export central : 1.46 s. Écart maximal après rechargement : 0.
Les embeddings déjà calculés sont réutilisés sans redécodage ni nouvelle inférence ViT. À l’inférence, une fenêtre exige un passage encodeur au lieu de trois ; la durée totale n’est pas nécessairement divisée par trois, car décodage et chargement ont leur propre coût.

Index vérifié, une fenêtre centrale exactement par morceau, colonnes et vecteurs identiques au sous-ensemble du cache parent. Scaler appris sur les seules fenêtres de train retenues. Mêmes grille C et règles de sélection. Rapports, matrices, modèles et prédictions vérifiés par empreintes ; métriques recalculées depuis les prédictions sauvegardées.

Les scores restent des résultats de développement sur FMA-small filtré. La validation sélectionne C et sert aux comparaisons ; aucun intervalle ni écart significatif sur de nouveaux artistes n’est revendiqué. Le test reste non évalué.

## Preuves et suite

Voir [le protocole de l’ablation](../docs/ABLATION_FENETRES.md).

- [Rapport trois fenêtres](../runs/vit-20261020T201146743988Z/report.json)
- [Rapport centrale](../runs/vit_central-20261021T061356952398Z/report.json)
- [Prédictions du diagnostic](../runs/vit_central-20261021T061356952398Z/diagnostic_three_train_central_eval.csv)
- [Comparaison vérifiable](VIT_WINDOWS_COMPARISON.json)

Prochain jalon : comparaison du bloc 6 et du bloc 12 du ViT, avec le même prétraitement, les trois fenêtres et la même grille. Benchmark préalable ; au maximum deux extractions supplémentaires pour les ablations, conformément au plan.
