# Bloc 6 + HPSS : comparaison de validation

Même cohorte : 6 299 morceaux de train et 780 de validation, trois fenêtres identiques. Le test reste non évalué. Configurations et grille fixées avant les résultats.

## Résultats

| Modèle | C | Macro-F1 validation | Accuracy | Macro-F1 train |
|---|---:|---:|---:|---:|
| vit_block6 | 0.1 | 0.5780 | 0.5885 | 0.6394 |
| vit_hpss | 1 | 0.5844 | 0.5974 | 0.6042 |
| vit | 0.1 | 0.5447 | 0.5538 | 0.6051 |
| vit_block6_hpss | 0.1 | 0.5975 | 0.6128 | 0.6352 |

## Différences candidat − référence

| Référence | Différence macro-F1 | Corrections | Régressions |
|---|---:|---:|---:|
| vit_block6 | +0.0195 | 59 | 40 |
| vit_hpss | +0.0131 | 56 | 44 |
| vit | +0.0528 | 87 | 41 |

Contre bloc 6 original, seuls les canaux changent ; contre bloc 12 HPSS, seule la profondeur change. Le bloc 12 original est un repère à deux facteurs différents. Ces effets incluent la sélection de C dans la même grille pour chaque représentation, sans garantir la généralisation.

## F1 par genre

| Genre | vit_block6 | vit_hpss | vit | vit_block6_hpss |
|---|---:|---:|---:|---:|
| Electronic | 0.5641 | 0.5541 | 0.5536 | 0.5862 |
| Experimental | 0.4808 | 0.5222 | 0.4571 | 0.5213 |
| Folk | 0.6935 | 0.7411 | 0.6702 | 0.7172 |
| Hip-Hop | 0.7610 | 0.6927 | 0.7343 | 0.7333 |
| Instrumental | 0.5405 | 0.5902 | 0.5028 | 0.5376 |
| International | 0.7254 | 0.7320 | 0.6598 | 0.7447 |
| Pop | 0.1558 | 0.1633 | 0.1098 | 0.1884 |
| Rock | 0.7033 | 0.6800 | 0.6701 | 0.7513 |

## Grille candidate

| C | Macro-F1 | Accuracy | Itérations |
|---:|---:|---:|---|
| 0.1 | 0.5975 | 0.6128 | [121] |
| 1 | 0.5906 | 0.6051 | [154] |
| 10 | 0.5891 | 0.6038 | [158] |

## Vérifications et limites

Extraction : 33.35 minutes dans l’invocation finale, 64 morceaux repris ; ce temps exclut leur calcul antérieur. Entraînement : 4.71 s. Export : écart maximal 0.

Caches, modèles, prédictions, cohorte et index sont vérifiés ; les métriques sont recalculées. Aucun avertissement d’ajustement. La validation sert à sélectionner C et à comparer les variantes : ces scores ne sont pas une estimation indépendante ni une preuve de significativité.

[Rapport numérique](VIT_BLOCK6_HPSS_COMPARISON.json) · [Protocole](../docs/COMBINAISON_BLOC6_HPSS.md)
