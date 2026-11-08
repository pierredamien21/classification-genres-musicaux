# Corrections et régressions de bloc 6 + HPSS

Analyse descriptive des 780 morceaux de validation, sans nouvel entraînement. Les étiquettes sont celles du manifeste ; aucune cause acoustique n’est déduite des seules prédictions. Le test reste non évalué.

## Face à vit_hpss

56 corrections, 44 régressions ; 258 morceaux restent mal classés par les deux modèles.

| Genre réel | Effectif | Corrections | Régressions | F1 référence | F1 candidat | FP référence → candidat |
|---|---:|---:|---:|---:|---:|---:|
| Electronic | 99 | 12 | 8 | 0.5541 | 0.5862 | 68 → 65 |
| Experimental | 100 | 9 | 7 | 0.5222 | 0.5213 | 50 → 56 |
| Folk | 99 | 3 | 5 | 0.7411 | 0.7172 | 25 → 28 |
| Hip-Hop | 100 | 6 | 0 | 0.6927 | 0.7333 | 34 → 33 |
| Instrumental | 93 | 8 | 12 | 0.5902 | 0.5376 | 36 → 43 |
| International | 98 | 4 | 5 | 0.7320 | 0.7447 | 25 → 20 |
| Pop | 94 | 5 | 4 | 0.1633 | 0.1884 | 41 → 31 |
| Rock | 97 | 9 | 3 | 0.6800 | 0.7513 | 35 → 26 |

FP : morceaux d’autres genres attribués à ce genre. Une F1 peut augmenter sans gain de rappel si les faux positifs diminuent.

Corrections réparties sur 42 artistes ; régressions sur 30 artistes. Maximum de régressions pour un artiste : 9. Ces comptes ne remplacent pas une estimation d’incertitude groupée par artiste.

## Face à vit_block6

59 corrections, 40 régressions ; 262 morceaux restent mal classés par les deux modèles.

| Genre réel | Effectif | Corrections | Régressions | F1 référence | F1 candidat | FP référence → candidat |
|---|---:|---:|---:|---:|---:|---:|
| Electronic | 99 | 9 | 7 | 0.5641 | 0.5862 | 69 → 65 |
| Experimental | 100 | 9 | 4 | 0.4808 | 0.5213 | 58 → 56 |
| Folk | 99 | 6 | 4 | 0.6935 | 0.7172 | 31 → 28 |
| Hip-Hop | 100 | 2 | 3 | 0.7610 | 0.7333 | 27 → 33 |
| Instrumental | 93 | 11 | 11 | 0.5405 | 0.5376 | 42 → 43 |
| International | 98 | 6 | 6 | 0.7254 | 0.7447 | 25 → 20 |
| Pop | 94 | 5 | 4 | 0.1558 | 0.1884 | 48 → 31 |
| Rock | 97 | 11 | 1 | 0.7033 | 0.7513 | 21 → 26 |

FP : morceaux d’autres genres attribués à ce genre. Une F1 peut augmenter sans gain de rappel si les faux positifs diminuent.

Corrections réparties sur 38 artistes ; régressions sur 29 artistes. Maximum de régressions pour un artiste : 6. Ces comptes ne remplacent pas une estimation d’incertitude groupée par artiste.

## Erreurs communes aux trois modèles

Bloc 6 HPSS, bloc 12 HPSS et bloc 6 original : Electronic 20; Experimental 38; Folk 21; Hip-Hop 20; Instrumental 24; International 21; Pop 74; Rock 20.

## Traçabilité et limites

Les modèles, caches, index et prédictions sont vérifiés par empreintes ; les métriques sont recalculées et les identifiants réconciliés avec le manifeste. La validation a guidé la sélection des variantes : aucun test de significativité ni intervalle de généralisation n’est présenté ici. Écoute subjective non réalisée.

Reproduction : `.\.venv\Scripts\python.exe scripts/analyze_block6_hpss_errors.py`.

- [Rapport numérique et empreintes](BLOCK6_HPSS_ERRORS.json)
- [Tous les morceaux corrigés ou dégradés](BLOCK6_HPSS_CHANGED_TRACKS.csv)
- [Décision et prochaines étapes](../docs/DECISIONS_EXPERIMENTALES.md)
