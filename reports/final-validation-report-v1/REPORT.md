# Évaluation — validation

Statistiques calculées ; contrôle indépendant du rapport restant à effectuer.

780 morceaux, 219 artistes, huit labels fixes.

| Système | Macro-F1 [IC 95 %] | Accuracy [IC 95 %] |
|---|---:|---:|
| descriptor | 0.4612 [0.3871 ; 0.5042] | 0.4756 [0.4035 ; 0.5476] |
| vit | 0.5447 [0.4542 ; 0.5810] | 0.5538 [0.4668 ; 0.6289] |
| fusion | 0.5219 [0.4353 ; 0.5577] | 0.5295 [0.4530 ; 0.5980] |
| vit_block6 | 0.5780 [0.4822 ; 0.6161] | 0.5885 [0.5043 ; 0.6598] |
| vit_hpss | 0.5844 [0.4759 ; 0.6283] | 0.5974 [0.5065 ; 0.6752] |
| vit_block6_hpss | 0.5975 [0.5034 ; 0.6352] | 0.6128 [0.5300 ; 0.6847] |
| cnn20_seed42 | 0.5464 [0.4450 ; 0.5855] | 0.5615 [0.4676 ; 0.6361] |
| cnn20_seed43 | 0.5471 [0.4525 ; 0.5897] | 0.5590 [0.4719 ; 0.6355] |
| cnn20_seed44 | 0.5287 [0.4410 ; 0.5666] | 0.5474 [0.4594 ; 0.6246] |
| majority | 0.0284 [0.0159 ; 0.0442] | 0.1282 [0.0681 ; 0.2148] |
| cnn20_mean | 0.5407 [0.4492 ; 0.5782] | 0.5560 [0.4689 ; 0.6289] |

Intervalles percentile ponctuels : 2 000 tirages appariés d’artistes,
conditionnels aux modèles fixés. Ils ne couvrent pas toute la variabilité
de l’apprentissage et ne sont pas des tests simultanés de supériorité.

La ligne cnn20_mean moyenne les métriques de trois graines, sans ensemble
de probabilités. Leur dispersion entre graines est enregistrée séparément.

## Comparaisons annoncées

| Différence gauche − droite | Δ macro-F1 [IC 95 %] | Δ accuracy [IC 95 %] |
|---|---:|---:|
| vit_block6_hpss - vit | +0.0528 [+0.0257 ; +0.0785] | +0.0590 [+0.0337 ; +0.0902] |
| fusion - vit | -0.0228 [-0.0490 ; +0.0073] | -0.0244 [-0.0551 ; +0.0097] |
| vit_block6_hpss - vit_hpss | +0.0131 [-0.0144 ; +0.0416] | +0.0154 [-0.0113 ; +0.0428] |
| vit_block6_hpss - vit_block6 | +0.0195 [-0.0098 ; +0.0436] | +0.0244 [+0.0012 ; +0.0497] |

## Candidat par genre

| Genre | Précision | Rappel | F1 | Effectif |
|---|---:|---:|---:|---:|
| Electronic | 0.5113 | 0.6869 | 0.5862 | 99 |
| Experimental | 0.4955 | 0.5500 | 0.5213 | 100 |
| Folk | 0.7172 | 0.7172 | 0.7172 | 99 |
| Hip-Hop | 0.7000 | 0.7700 | 0.7333 | 100 |
| Instrumental | 0.5376 | 0.5376 | 0.5376 | 93 |
| International | 0.7778 | 0.7143 | 0.7447 | 98 |
| Pop | 0.2955 | 0.1383 | 0.1884 | 94 |
| Rock | 0.7400 | 0.7629 | 0.7513 | 97 |

## Coût et limites

Les durées par appel proviennent des prédictions sauvegardées : elles incluent
le prétraitement et l’inférence, mais pas tous les contrôles de campagne.
La durée de la dernière invocation n’est pas une somme de reprises.

Limites conservées : huit genres fermés, pas de gospel ni de détection du
genre inconnu, scores non calibrés, fragilité au bruit, écoute subjective absente.

**Validation utilisée pour sélectionner les modèles : aucune performance indépendante
de généralisation n’est revendiquée dans ce rapport.**

Les matrices de confusion sont dans `confusions.csv`, les détails par classe
dans `per_class.csv`, les preuves dans `report.json` et les tirages dans
`bootstrap_draws.npz`. Les lignes sont les vrais genres, les colonnes les prédictions.
