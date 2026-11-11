# ViT : original/harmonique/percussif contre log-Mel répété

Ablation au bloc 12 gelé, sur 6 299 morceaux de train et 780 de validation. Mêmes trois fenêtres, checkpoint, 192 caractéristiques et grille logistique.

## Résultats par morceau

| Canaux / profondeur | C | Macro-F1 validation | Accuracy | Macro-F1 train |
|---|---:|---:|---:|---:|
| Original répété / bloc 12 | 0.1 | 0.5447 | 0.5538 | 0.6051 |
| Original + H + P / bloc 12 | 1 | 0.5844 | 0.5974 | 0.6042 |
| Original répété / bloc 6 (contexte) | 0.1 | 0.5780 | 0.5885 | 0.6394 |

Écart HPSS − original, à profondeur identique : **+0.0397** en macro-F1.
Cette variante HPSS améliore la référence bloc 12 sur validation.

Le bloc 6 est affiché pour situer les résultats du projet. Sa comparaison avec HPSS change à la fois profondeur et canaux : elle ne mesure pas l’effet isolé de HPSS.

## Grille complète

| C | Original bloc 12 | HPSS bloc 12 |
|---:|---:|---:|
| 0.1 | 0.5447 | 0.5835 |
| 1 | 0.5438 | 0.5844 |
| 10 | 0.5438 | 0.5844 |

## F1 par genre

| Genre | Effectif | Original bloc 12 | HPSS bloc 12 |
|---|---:|---:|---:|
| Electronic | 99 | 0.5536 | 0.5541 |
| Experimental | 100 | 0.4571 | 0.5222 |
| Folk | 99 | 0.6702 | 0.7411 |
| Hip-Hop | 100 | 0.7343 | 0.6927 |
| Instrumental | 93 | 0.5028 | 0.5902 |
| International | 98 | 0.6598 | 0.7320 |
| Pop | 94 | 0.1098 | 0.1633 |
| Rock | 97 | 0.6701 | 0.6800 |

## Erreurs appariées et confusion

- HPSS correct, original incorrect : 77 morceaux.
- Original correct, HPSS incorrect : 43 morceaux.

Confusion HPSS : lignes réelles, colonnes prédites.

| Réel / prédit | Electronic | Experimental | Folk | Hip-Hop | Instrumental | International | Pop | Rock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Electronic | 64 | 3 | 3 | 10 | 13 | 1 | 4 | 1 |
| Experimental | 15 | 53 | 7 | 3 | 8 | 2 | 4 | 8 |
| Folk | 0 | 2 | 73 | 3 | 4 | 2 | 14 | 1 |
| Hip-Hop | 19 | 2 | 1 | 71 | 0 | 5 | 1 | 1 |
| Instrumental | 3 | 14 | 4 | 0 | 54 | 1 | 9 | 8 |
| International | 11 | 3 | 1 | 3 | 5 | 71 | 2 | 2 |
| Pop | 16 | 20 | 6 | 13 | 5 | 8 | 12 | 14 |
| Rock | 4 | 6 | 3 | 2 | 1 | 6 | 7 | 68 |

## Vérifications et coût

Empreintes des rapports, matrices, modèles et prédictions vérifiées ; index identiques octet pour octet ; métriques recalculées depuis les probabilités exportées. Le scaler est appris exclusivement sur train et les trois ajustements logistiques ont convergé.

- Écart maximal canal original / ancien log-Mel : 0 dB.
- Écart maximal vecteur original répété / ancien cache, sur 64 fenêtres de train : 0.
- Médiane HPSS + log-Mel + ViT : 0.4949 s par fenêtre (un fil).
- Médiane décodage : 0.1017 s par morceau de benchmark.
- Durée écoulée enregistrée pour l’extraction : 311.32 min, 4 processus.
- Grille logistique et export : 5.78 s.
- Écart de probabilités après rechargement du classifieur : 0.

Les durées dépendent de la charge du poste. Le bloc 6 réutilisait des log-Mel en cache ; la nouvelle extraction redécode les MP3 et recalcule le spectre nécessaire à HPSS. Les durées globales ne mesurent donc pas seulement le coût relatif des architectures.
Les horodatages des caches comportent 3 intervalle(s) de plus de deux minutes sans nouvelle sauvegarde, avec un maximum de 216.76 minutes. La cause de ces grands écarts n’est pas enregistrée. La durée globale ne doit pas être présentée comme une mesure contrôlée de temps CPU continu ; le benchmark par fenêtre est plus informatif ici.

## Limites et suite

Le ViT est préentraîné sur des images RGB : l’ordre original/H/P est une convention de transfert, sans apprentissage d’un agencement audio spécifique. HPSS distingue des structures durables et transitoires, sans isoler parfaitement instruments ou voix. Les résultats concernent cette configuration ; ils ne prouvent pas une amélioration générale de la compréhension de l’harmonie.

La validation a servi au choix de C et aux comparaisons ; aucun gain statistiquement significatif ni score final indépendant n’est revendiqué. Les deux extractions supplémentaires prévues sont consommées. Prochaine étape : consolider le protocole final, réaliser les trois graines CNN prévues et verrouiller les choix avant le test. Le test reste non évalué.

## Reproduction et preuves

Voir [le protocole et sa source HPSS](../docs/archive/ABLATION_HPSS.md).

- [Rapport original](../runs/vit-20261020T201146743988Z/report.json)
- [Rapport HPSS](../runs/vit_hpss-20261021T115423153046Z/report.json)
- [Comparaison vérifiable](VIT_HPSS_COMPARISON.json)
- [Benchmark](VIT_HPSS_BENCHMARK.json)
