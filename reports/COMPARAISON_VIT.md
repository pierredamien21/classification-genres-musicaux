# ViT gelé et descripteurs musicaux — comparaison sur validation

Même sélection filtrée de FMA-small : 6 299 morceaux d’entraînement, 780 de validation ; 21 237 fenêtres identiques. Le jeu de test n’est ni extrait ni évalué pour cette expérience.

## Résultats par morceau

| Système | Dimensions | C choisi | Macro-F1 validation | Accuracy validation | Macro-F1 train |
|---|---:|---:|---:|---:|---:|
| Majoritaire (Experimental) | — | — | 0.0284 | 0.1282 | — |
| Descripteurs | 125 | 0.1 | 0.4612 | 0.4756 | 0.5446 |
| ViT gelé | 192 | 0.1 | 0.5447 | 0.5538 | 0.6051 |

Écart ViT − descripteurs : **+0.0835** en macro-F1.

## Grille ViT

| C | Macro-F1 validation | Accuracy | Itérations |
|---:|---:|---:|---:|
| 0.1 | 0.5447 | 0.5538 | 99 |
| 1 | 0.5438 | 0.5526 | 110 |
| 10 | 0.5438 | 0.5526 | 109 |

## F1 par genre

| Genre | Effectif validation | Descripteurs | ViT |
|---|---:|---:|---:|
| Electronic | 99 | 0.4375 | 0.5536 |
| Experimental | 100 | 0.4519 | 0.4571 |
| Folk | 99 | 0.7022 | 0.6702 |
| Hip-Hop | 100 | 0.5140 | 0.7343 |
| Instrumental | 93 | 0.3253 | 0.5028 |
| International | 98 | 0.4848 | 0.6598 |
| Pop | 94 | 0.1242 | 0.1098 |
| Rock | 97 | 0.6497 | 0.6701 |

## Erreurs appariées

- Les deux systèmes corrects : 287 morceaux.
- Descripteurs seuls corrects : 84.
- ViT seul correct : 145.
- Les deux systèmes incorrects : 264.

Ces désaccords motivent l’expérience de fusion. Ils ne garantissent pas qu’un classifieur saura combiner les représentations et ne constituent pas un test de significativité.

## Confusion ViT

Lignes : genre réel ; colonnes : genre prédit.

| Réel / prédit | Electronic | Experimental | Folk | Hip-Hop | Instrumental | International | Pop | Rock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Electronic | 62 | 7 | 1 | 10 | 9 | 4 | 5 | 1 |
| Experimental | 19 | 48 | 5 | 2 | 10 | 4 | 3 | 9 |
| Folk | 1 | 3 | 63 | 3 | 7 | 3 | 18 | 1 |
| Hip-Hop | 13 | 2 | 1 | 76 | 0 | 5 | 2 | 1 |
| Instrumental | 3 | 16 | 6 | 0 | 45 | 3 | 16 | 4 |
| International | 10 | 4 | 4 | 3 | 6 | 64 | 5 | 2 |
| Pop | 14 | 23 | 7 | 11 | 7 | 9 | 9 | 14 |
| Rock | 3 | 7 | 2 | 2 | 2 | 4 | 12 | 65 |

## Coût et contrôles

Extraction descripteurs : 11.68 min ; ViT : 9.49 min.
Durées observées sur ce poste, quatre processus CPU ; elles dépendent de la charge et ne sont pas des mesures contrôlées de vitesse relative.
Entraînement et export ViT : 3.80 s. Écart maximal après rechargement : 0.
Empreintes vérifiées : rapports, matrices, index, modèles et prédictions. Index de fenêtres identiques octet pour octet ; même grille C et même protocole. Métriques recalculées depuis les probabilités exportées. Standardisation ajustée sur train uniquement.

## Interprétation et limites

Le ViT est préentraîné sur des images et reste intégralement gelé. Seule la tête logistique est entraînée sur FMA. Le redimensionnement carré compresse le temps ; la plage dB fixe n’a pas été optimisée. Cette expérience mesure une référence de transfert, pas la performance maximale d’un Transformer audio.

Le choix de C utilise la validation : ces scores servent au développement et ne sont pas une estimation finale indépendante. Aucun intervalle de confiance ni gain sur test n’est revendiqué. Les catégories sont les huit genres FMA ; le gospel n’est pas une classe de cette version.

## Reproduction et preuves

Voir [le protocole ViT](../docs/VIT_GELE.md).

- [Rapport descripteurs](../runs/descriptors-20261020T195511254895Z/report.json)
- [Rapport ViT](../runs/vit-20261020T201146743988Z/report.json)
- [Comparaison vérifiable](VIT_COMPARISON.json)
- [Benchmark de 64 fenêtres](VIT_BENCHMARK.json)
