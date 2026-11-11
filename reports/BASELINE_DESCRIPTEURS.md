# Première référence : descripteurs musicaux

Expérience du 2026-10-20 sur le manifeste filtré `de1e43489ae9f4d335396db98dc7cd050ed44a230d6a5e0df3fc8417310c7171`.

**Résultats de validation uniquement : le jeu de test n’a pas été évalué.**

| Système | Macro-F1 par morceau | Accuracy |
|---|---:|---:|
| Genre majoritaire (Experimental) | 0.0284 | 0.1282 |
| Descripteurs + régression logistique | **0.4612** | **0.4756** |

Évaluation sur 780 morceaux de validation. Entraînement sur 6299 morceaux (18897 fenêtres). Les probabilités des fenêtres sont moyennées par morceau avant les métriques.

## Ce que représente chaque fenêtre

| Famille | Dimensions | Informations résumées |
|---|---:|---|
| Timbre et énergie | 60 | 20 MFCC, centroïde, largeur et contraste spectraux, RMS ; moyennes et écarts-types |
| Harmonie | 24 | Moyennes et écarts-types du chroma à 12 classes de hauteur |
| Rythme | 41 | Autocorrélation du flux spectral sur une grille 40–240 BPM |

Mono 16 kHz, fenêtres à 0, 10 et 20 secondes, FFT 1 024, pas 256. Les statistiques utilisent les trames complètes du signal réel ; une seule trame complétée est autorisée si le signal est plus court que la FFT. Le padding final n’ajoute donc pas de faux silence aux moyennes. Les paramètres exacts sont conservés dans l’expérience.

Le chroma n’est pas une transcription d’accords ; le profil rythmique peut contenir plusieurs périodicités et ne constitue pas une mesure certaine du tempo.

## Sélection du classifieur

StandardScaler est ajusté uniquement sur les fenêtres de training. Même famille de classifieur pour les trois valeurs de C prévues ; sélection par macro-F1 de validation et préférence au plus petit C en cas d’égalité.

| C | Macro-F1 validation | Accuracy validation | Itérations |
|---:|---:|---:|---|
| 0.1 | 0.4612 | 0.4756 | [101] |
| 1.0 | 0.4555 | 0.4705 | [126] |
| 10.0 | 0.4573 | 0.4718 | [139] |

**C retenu : 0.1.** Les modèles ont convergé. Macro-F1 d’entraînement du modèle retenu : 0.5446. L’écart avec la validation est descriptif ; il ne remplace pas l’évaluation finale.

## Résultats par genre sur validation

| Genre | Précision | Rappel | F1 | Morceaux |
|---|---:|---:|---:|---:|
| Electronic | 0.392 | 0.495 | 0.438 | 99 |
| Experimental | 0.435 | 0.470 | 0.452 | 100 |
| Folk | 0.627 | 0.798 | 0.702 | 99 |
| Hip-Hop | 0.482 | 0.550 | 0.514 | 100 |
| Instrumental | 0.370 | 0.290 | 0.325 | 93 |
| International | 0.597 | 0.408 | 0.485 | 98 |
| Pop | 0.149 | 0.106 | 0.124 | 94 |
| Rock | 0.640 | 0.660 | 0.650 | 97 |

### Matrice de confusion

Lignes : genre réel ; colonnes : genre prédit. Comptages par morceau.

| Réel / Prédit | Electronic | Experimental | Folk | Hip-Hop | Instrumental | International | Pop | Rock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Electronic | 49 | 6 | 3 | 19 | 10 | 5 | 6 | 1 |
| Experimental | 15 | 47 | 6 | 3 | 16 | 1 | 2 | 10 |
| Folk | 1 | 0 | 79 | 3 | 5 | 2 | 5 | 4 |
| Hip-Hop | 25 | 1 | 2 | 55 | 0 | 10 | 7 | 0 |
| Instrumental | 6 | 19 | 10 | 1 | 27 | 2 | 20 | 8 |
| International | 11 | 12 | 14 | 11 | 3 | 40 | 5 | 2 |
| Pop | 17 | 18 | 8 | 15 | 10 | 5 | 10 | 11 |
| Rock | 1 | 5 | 4 | 7 | 2 | 2 | 12 | 64 |

## Calcul, export et reproduction

- Caractéristiques : 21,237 fenêtres × 125 valeurs, toutes finies ; extraction en 11.7 minutes sur CPU.
- Entraînement, sélection et export : 3.6 secondes.
- Export numérique rechargé : différence maximale de probabilité 0.00e+00 par rapport à scikit-learn.
- Provenance et résultats complets (`runs/descriptors-20261020T195511254895Z/report.json`, non versionné)
- Prédictions de validation (`runs/descriptors-20261020T195511254895Z/validation_predictions.csv`, non versionné)
- Configuration du modèle (`runs/descriptors-20261020T195511254895Z/model.json`, non versionné)
- Export numérique sans pickle (`runs/descriptors-20261020T195511254895Z/model.npz`, non versionné)
- Statistiques des caractéristiques sur train (`runs/descriptors-20261020T195511254895Z/train_feature_statistics.json`, non versionné)

## Limites et suite

Le meilleur C est choisi sur cette validation : son score ne constitue donc pas une estimation finale indépendante. Aucun intervalle de confiance sur test n’est calculé. Ce sous-ensemble filtré de FMA-small doit être explicitement distingué du benchmark non filtré. Aucun gain de la fusion, du CNN ou du ViT n’est encore démontré.

Prochaine comparaison : extraire le ViT gelé sur les mêmes fenêtres et les mêmes morceaux, puis appliquer la même grille de classifieur. La fusion et les ablations suivront ; le test restera réservé au protocole final.
