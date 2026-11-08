# Évaluation — test

Statistiques calculées ; contrôle indépendant du rapport restant à effectuer.

778 morceaux, 202 artistes, huit labels fixes.

| Système | Macro-F1 [IC 95 %] | Accuracy [IC 95 %] |
|---|---:|---:|
| descriptor | 0.3892 [0.3011 ; 0.4260] | 0.3959 [0.3058 ; 0.5072] |
| vit | 0.4641 [0.3761 ; 0.5073] | 0.4756 [0.3803 ; 0.5851] |
| fusion | 0.4791 [0.3892 ; 0.5142] | 0.4871 [0.3937 ; 0.6002] |
| vit_block6 | 0.4883 [0.3991 ; 0.5294] | 0.4949 [0.4078 ; 0.6047] |
| vit_hpss | 0.4694 [0.3844 ; 0.5108] | 0.4794 [0.3969 ; 0.5773] |
| vit_block6_hpss | 0.5035 [0.4133 ; 0.5449] | 0.5116 [0.4280 ; 0.6129] |
| cnn20_seed42 | 0.4539 [0.3643 ; 0.5003] | 0.4602 [0.3719 ; 0.5714] |
| cnn20_seed43 | 0.4424 [0.3536 ; 0.4976] | 0.4640 [0.3640 ; 0.5794] |
| cnn20_seed44 | 0.3995 [0.3069 ; 0.4502] | 0.4165 [0.3119 ; 0.5437] |
| majority | 0.0277 [0.0157 ; 0.0437] | 0.1247 [0.0672 ; 0.2120] |
| cnn20_mean | 0.4320 [0.3451 ; 0.4796] | 0.4469 [0.3520 ; 0.5630] |

Intervalles percentile ponctuels : 2 000 tirages appariés d’artistes,
conditionnels aux modèles fixés. Ils ne couvrent pas toute la variabilité
de l’apprentissage et ne sont pas des tests simultanés de supériorité.

La ligne cnn20_mean moyenne les métriques de trois graines, sans ensemble
de probabilités. Leur dispersion entre graines est enregistrée séparément.

## Comparaisons annoncées

| Différence gauche − droite | Δ macro-F1 [IC 95 %] | Δ accuracy [IC 95 %] |
|---|---:|---:|
| vit_block6_hpss - vit | +0.0394 [+0.0054 ; +0.0715] | +0.0360 [-0.0013 ; +0.0711] |
| fusion - vit | +0.0150 [-0.0277 ; +0.0508] | +0.0116 [-0.0321 ; +0.0476] |
| vit_block6_hpss - vit_hpss | +0.0341 [-0.0006 ; +0.0658] | +0.0321 [-0.0064 ; +0.0659] |
| vit_block6_hpss - vit_block6 | +0.0152 [-0.0112 ; +0.0406] | +0.0167 [-0.0098 ; +0.0423] |

## Candidat par genre

| Genre | Précision | Rappel | F1 | Effectif |
|---|---:|---:|---:|---:|
| Electronic | 0.5776 | 0.7053 | 0.6351 | 95 |
| Experimental | 0.4167 | 0.3093 | 0.3550 | 97 |
| Folk | 0.3545 | 0.3900 | 0.3714 | 100 |
| Hip-Hop | 0.7308 | 0.7755 | 0.7525 | 98 |
| Instrumental | 0.4737 | 0.3750 | 0.4186 | 96 |
| International | 0.5093 | 0.5500 | 0.5288 | 100 |
| Pop | 0.3372 | 0.3152 | 0.3258 | 92 |
| Rock | 0.6226 | 0.6600 | 0.6408 | 100 |

## Coût et limites

Les durées par appel proviennent des prédictions sauvegardées : elles incluent
le prétraitement et l’inférence, mais pas tous les contrôles de campagne.
La durée de la dernière invocation n’est pas une somme de reprises.

Limites conservées : huit genres fermés, pas de gospel ni de détection du
genre inconnu, scores non calibrés, fragilité au bruit, écoute subjective absente.

Les matrices de confusion sont dans `confusions.csv`, les détails par classe
dans `per_class.csv`, les preuves dans `report.json` et les tirages dans
`bootstrap_draws.npz`. Les lignes sont les vrais genres, les colonnes les prédictions.
