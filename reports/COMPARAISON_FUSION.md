# Fusion ViT et descripteurs — résultats et ablations

Expériences sur les mêmes 6 299 morceaux d’entraînement et 780 morceaux de validation, avec 21 237 fenêtres alignées. Les deux ablations retirent séparément les familles rythme et harmonie prévues dans le plan. Aucun nouvel audio n’a été décodé.

## Comparaison principale

| Système | Dimensions | C choisi | Macro-F1 validation | Accuracy | Macro-F1 train | Écart au ViT |
|---|---:|---:|---:|---:|---:|---:|
| Descripteurs | 125 | 0.1 | 0.4612 | 0.4756 | 0.5446 | -0.0835 |
| ViT seul | 192 | 0.1 | 0.5447 | 0.5538 | 0.6051 | +0.0000 |
| Fusion complète | 317 | 0.1 | 0.5219 | 0.5295 | 0.6678 | -0.0228 |
| Fusion sans rythme | 276 | 1 | 0.5387 | 0.5487 | 0.6577 | -0.0060 |
| Fusion sans harmonie | 293 | 0.1 | 0.5282 | 0.5385 | 0.6582 | -0.0165 |

## Grille complète et comparaison à C fixe

Chaque système dispose du même budget : trois valeurs de C, sélection sur la macro-F1 par morceau. La colonne C = 0,1 permet aussi de comparer les retraits de familles au même niveau de régularisation. Aucun essai n’est omis.

| Système | Macro-F1 C = 0,1 | Macro-F1 C = 1 | Macro-F1 C = 10 |
|---|---:|---:|---:|
| Descripteurs | 0.4612 | 0.4555 | 0.4573 |
| ViT seul | 0.5447 | 0.5438 | 0.5438 |
| Fusion complète | 0.5219 | 0.5184 | 0.5185 |
| Fusion sans rythme | 0.5383 | 0.5387 | 0.5374 |
| Fusion sans harmonie | 0.5282 | 0.5173 | 0.5193 |

## F1 par genre

| Genre | Effectif | Descripteurs | ViT seul | Fusion complète | Fusion sans rythme | Fusion sans harmonie |
|---|---:|---:|---:|---:|---:|---:|
| Electronic | 99 | 0.4375 | 0.5536 | 0.5000 | 0.5388 | 0.5088 |
| Experimental | 100 | 0.4519 | 0.4571 | 0.4623 | 0.4876 | 0.4752 |
| Folk | 99 | 0.7022 | 0.6702 | 0.7228 | 0.6935 | 0.7136 |
| Hip-Hop | 100 | 0.5140 | 0.7343 | 0.6634 | 0.6981 | 0.6699 |
| Instrumental | 93 | 0.3253 | 0.5028 | 0.4505 | 0.4494 | 0.4444 |
| International | 98 | 0.4848 | 0.6598 | 0.6277 | 0.6943 | 0.6425 |
| Pop | 94 | 0.1242 | 0.1098 | 0.1065 | 0.0952 | 0.1118 |
| Rock | 97 | 0.6497 | 0.6701 | 0.6417 | 0.6526 | 0.6597 |

## Erreurs par rapport au ViT

| Système | Erreurs du ViT corrigées | Bonnes prédictions du ViT perdues |
|---|---:|---:|
| Fusion complète | 50 | 69 |
| Fusion sans rythme | 55 | 59 |
| Fusion sans harmonie | 57 | 69 |

## Coût et traçabilité

| Expérience | Assemblage cache (s) | Entraînement et export (s) | Écart export maximal |
|---|---:|---:|---:|
| Fusion complète | 0.292 | 6.751 | 0 |
| Fusion sans rythme | 0.280 | 5.652 | 0 |
| Fusion sans harmonie | 0.288 | 6.452 | 0 |

Les temps d’assemblage réutilisent les caches existants ; ils n’incluent pas les 9,49 minutes d’extraction ViT ni les 11,68 minutes de descripteurs. À l’inférence sur un nouveau morceau, une fusion doit calculer les deux représentations.

Vérifications : empreintes des caches, des modèles et des prédictions ; index identiques ; valeurs fusionnées égales à leurs colonnes sources ; même grille ; métriques recalculées depuis les prédictions exportées. Standardisation ajustée sur train, neuf ajustements de fusion convergés et exports numériques rechargés.

## Interprétation

La concaténation complète n’améliore pas le ViT seul dans ce protocole. Retirer le rythme ou l’harmonie relève son score, mais aucune de ces variantes ne dépasse le ViT. L’hypothèse d’un gain par ajout des descripteurs n’est donc pas soutenue par ces essais.

Ce constat ne signifie pas que rythme ou harmonie seraient inutiles pour reconnaître les genres. Il concerne ces résumés statistiques, cette représentation ViT, ce classifieur et cette sélection de données. Redondance, interactions, régularisation et estimation sur un échantillon fini sont des explications possibles, pas des causes démontrées.

La standardisation par dimension ne donne pas le même poids global à chaque famille : elles contiennent des nombres de dimensions différents. La concaténation augmente aussi la capacité du classifieur. Les ablations mesurent un retrait de famille dans ce système, pas une importance musicale universelle.

La validation a servi à choisir C et à comparer plusieurs systèmes. Les écarts observés ne prouvent pas une différence significative sur de nouveaux artistes. Le test reste réservé au protocole final ; aucune nouvelle recherche de paramètres n’est ajoutée à cette étape pour renverser le résultat.

## Suite et reproduction

Conserver le ViT seul comme meilleure référence actuelle selon la macro-F1 de validation. Prochain jalon : benchmark du petit CNN prévu au plan, puis entraînement dans le budget CPU si sa faisabilité est confirmée. Le choix final sera verrouillé avant le test.

Voir [le protocole de fusion](../docs/FUSION.md) et [les résultats vérifiables](FUSION_COMPARISON.json).

- [Descripteurs : rapport d’expérience](../runs/descriptors-20261020T195511254895Z/report.json)
- [ViT seul : rapport d’expérience](../runs/vit-20261020T201146743988Z/report.json)
- [Fusion complète : rapport d’expérience](../runs/fusion-20261020T201803253265Z/report.json)
- [Fusion sans rythme : rapport d’expérience](../runs/fusion_no_rhythm-20261020T201813274845Z/report.json)
- [Fusion sans harmonie : rapport d’expérience](../runs/fusion_no_harmony-20261020T201822268410Z/report.json)
