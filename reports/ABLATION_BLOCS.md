# ViT : bloc 6 contre bloc 12

Ablation sur les mêmes 6 299 morceaux de train et 780 de validation, trois fenêtres par morceau, 192 caractéristiques et grille C identique. Les poids du ViT restent gelés.

## Résultats par morceau

| Représentation | C retenu | Macro-F1 validation | Accuracy validation | Macro-F1 train |
|---|---:|---:|---:|---:|
| Bloc 12 (référence) | 0.1 | 0.5447 | 0.5538 | 0.6051 |
| Bloc 6 | 0.1 | 0.5780 | 0.5885 | 0.6394 |

Écart bloc 6 − bloc 12 : **+0.0333** en macro-F1.
Le **bloc 6** est retenu comme référence entre ces deux variantes selon le critère principal de validation.

## Grille complète

| C | Macro-F1 bloc 12 | Macro-F1 bloc 6 |
|---:|---:|---:|
| 0.1 | 0.5447 | 0.5780 |
| 1 | 0.5438 | 0.5738 |
| 10 | 0.5438 | 0.5760 |

## F1 par genre

| Genre | Effectif | Bloc 12 | Bloc 6 |
|---|---:|---:|---:|
| Electronic | 99 | 0.5536 | 0.5641 |
| Experimental | 100 | 0.4571 | 0.4808 |
| Folk | 99 | 0.6702 | 0.6935 |
| Hip-Hop | 100 | 0.7343 | 0.7610 |
| Instrumental | 93 | 0.5028 | 0.5405 |
| International | 98 | 0.6598 | 0.7254 |
| Pop | 94 | 0.1098 | 0.1558 |
| Rock | 97 | 0.6701 | 0.7033 |

## Erreurs appariées

- Bloc 6 correct, bloc 12 incorrect : 74 morceaux.
- Bloc 12 correct, bloc 6 incorrect : 47 morceaux.

## Confusion du bloc 6

Lignes : genre réel ; colonnes : genre prédit.

| Réel / prédit | Electronic | Experimental | Folk | Hip-Hop | Instrumental | International | Pop | Rock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Electronic | 66 | 10 | 1 | 9 | 7 | 2 | 4 | 0 |
| Experimental | 18 | 50 | 4 | 3 | 14 | 3 | 0 | 8 |
| Folk | 3 | 1 | 69 | 1 | 6 | 4 | 14 | 1 |
| Hip-Hop | 14 | 1 | 1 | 78 | 0 | 4 | 1 | 1 |
| Instrumental | 6 | 16 | 5 | 0 | 50 | 2 | 12 | 2 |
| International | 7 | 3 | 3 | 2 | 7 | 70 | 4 | 2 |
| Pop | 19 | 23 | 12 | 10 | 5 | 6 | 12 | 7 |
| Rock | 2 | 4 | 5 | 2 | 3 | 4 | 13 | 64 |

## Protocole et vérifications

Le même calcul de patches, les mêmes positions, six premiers blocs identiques et la même LayerNorm finale gelée du checkpoint sont utilisés. La variante s’arrête après le bloc 6 avant cette LayerNorm ; la référence traverse les douze blocs. Dans les deux cas, les tokens de préfixe sont exclus avant la moyenne des patches. Aucun poids n’est réajusté.

La LayerNorm finale a été apprise pour la sortie du bloc 12 pendant le préentraînement. Son emploi au bloc 6 est une convention explicite de cette ablation, pas une normalisation spécifiquement apprise pour la couche intermédiaire.

Sur 64 fenêtres de train, le chemin log-Mel en cache reproduit les vecteurs du bloc 12 avec un écart maximal de 0. Le test unitaire compare aussi la sortie tronquée à une capture directe après le sixième bloc du réseau complet, suivie de la même LayerNorm.

Rapports, matrices, modèles, prédictions et benchmark contrôlés par empreintes. Index identiques octet pour octet ; mêmes labels, partitions et budgets de recherche. Métriques recalculées depuis les prédictions sauvegardées. Scalers ajustés uniquement sur les fenêtres de train.

## Coût observé

- Médiane bloc 6 : 0.0421 s par fenêtre, après chauffe, un fil CPU.
- Extraction complète : 3.97 min, 4 processus.
- Grille logistique et export : 5.88 s.
- Écart maximal après rechargement du classifieur : 0.

L’extraction réutilise les log-Mel déjà calculés pour le CNN. Sa durée ne peut pas être comparée directement à celle du bloc 12, qui incluait le décodage des MP3 et le calcul des spectrogrammes. Une des deux extractions supplémentaires prévues pour les ablations est consommée ; l’ablation centrale n’en a consommé aucune.

## Limites et suite

Les résultats concernent ce checkpoint, ce pooling, cette convention de normalisation et le sous-ensemble filtré FMA-small. Ils ne déterminent pas la meilleure couche pour tout modèle audio. La validation choisit C et sert aux comparaisons : aucun score final indépendant ni écart significatif sur de nouveaux artistes n’est revendiqué. Le test reste non évalué.

Prochaine ablation prévue : canaux log-Mel original/harmonique/percussif contre répétition du log-Mel, sur la référence bloc 12 du plan pour ne changer qu’un facteur. Benchmark préalable et dernière extraction supplémentaire autorisée par le plan.

## Reproduction et preuves

Voir [le protocole](../docs/archive/ABLATION_BLOCS.md).

- [Rapport bloc 12](../runs/vit-20261020T201146743988Z/report.json)
- [Rapport bloc 6](../runs/vit_block6-20261021T062701855205Z/report.json)
- [Comparaison vérifiable](VIT_BLOCKS_COMPARISON.json)
- [Benchmark](VIT_BLOCK6_BENCHMARK.json)
