# Erreurs de validation : Pop, Hip-Hop et désaccords

Analyse automatisée terminée ; revue d’écoute préparée mais **non réalisée**. Les métriques et prédictions sont vérifiées sur les mêmes 780 morceaux. Test non évalué.

## Précision et rappel

| Modèle | Genre | Support | Artistes | Précision | Rappel | F1 |
|---|---|---:|---:|---:|---:|---:|
| ViT bloc 12 original | Pop | 94 | 26 | 0.1286 | 0.0957 | 0.1098 |
| ViT bloc 12 original | Hip-Hop | 100 | 17 | 0.7103 | 0.7600 | 0.7343 |
| ViT bloc 6 original | Pop | 94 | 26 | 0.2000 | 0.1277 | 0.1558 |
| ViT bloc 6 original | Hip-Hop | 100 | 17 | 0.7429 | 0.7800 | 0.7610 |
| ViT bloc 12 HPSS | Pop | 94 | 26 | 0.2264 | 0.1277 | 0.1633 |
| ViT bloc 12 HPSS | Hip-Hop | 100 | 17 | 0.6762 | 0.7100 | 0.6927 |
| Descripteurs | Pop | 94 | 26 | 0.1493 | 0.1064 | 0.1242 |
| Descripteurs | Hip-Hop | 100 | 17 | 0.4825 | 0.5500 | 0.5140 |
| CNN prolongé, graine 42 | Pop | 94 | 26 | 0.1270 | 0.0851 | 0.1019 |
| CNN prolongé, graine 42 | Hip-Hop | 100 | 17 | 0.6857 | 0.7200 | 0.7024 |
| CNN prolongé, graine 43 | Pop | 94 | 26 | 0.1525 | 0.0957 | 0.1176 |
| CNN prolongé, graine 43 | Hip-Hop | 100 | 17 | 0.6854 | 0.6100 | 0.6455 |
| CNN prolongé, graine 44 | Pop | 94 | 26 | 0.0943 | 0.0532 | 0.0680 |
| CNN prolongé, graine 44 | Hip-Hop | 100 | 17 | 0.6228 | 0.7100 | 0.6636 |

CNN : trois réseaux distincts sont affichés ; aucun ensemble ni meilleure graine sélectionnée.

## Ce que HPSS corrige ou dégrade face au bloc 12 original

| Ensemble | Les deux corrects | Corrigés par HPSS | Perdus avec HPSS | Les deux faux | Désaccords |
|---|---:|---:|---:|---:|---:|
| Tous | 389 | 77 | 43 | 271 | 187 |
| Electronic | 57 | 7 | 5 | 30 | 22 |
| Experimental | 44 | 9 | 4 | 43 | 21 |
| Folk | 60 | 13 | 3 | 23 | 20 |
| Hip-Hop | 68 | 3 | 8 | 21 | 14 |
| Instrumental | 38 | 16 | 7 | 32 | 33 |
| International | 59 | 12 | 5 | 22 | 24 |
| Pop | 4 | 8 | 5 | 77 | 32 |
| Rock | 59 | 9 | 6 | 23 | 21 |

Deux modèles peuvent se tromper avec des classes différentes : un désaccord n’est pas nécessairement une correction.

## Vers quels genres vont les morceaux Pop et Hip-Hop sous HPSS ?

| Genre réel | Electronic | Experimental | Folk | Hip-Hop | Instrumental | International | Pop | Rock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pop | 16 | 20 | 6 | 13 | 5 | 8 | 12 | 14 |
| Hip-Hop | 19 | 2 | 1 | 71 | 0 | 5 | 1 | 1 |

*Figure non versionnée : `reports/figures/validation_confusions.png`*

Figure PDF (`reports/figures/validation_confusions.pdf`, non versionné). Couleurs normalisées par genre réel, nombres bruts dans les cellules.

## Échantillon de revue

**19 extraits, 19 artistes distincts, aucune annotation d’écoute réalisée.**

| Catégorie | Candidats | Sélectionnés | Manquants au quota |
|---|---:|---:|---:|
| Hip-Hop retrouvée avec HPSS | 3 | 2 | 1 |
| Pop manquée par HPSS, retrouvée au bloc 6 | 6 | 3 | 0 |
| Hip-Hop correcte avant HPSS, perdue après | 8 | 2 | 1 |
| Pop correctement reconnue par HPSS | 12 | 3 | 0 |
| Hip-Hop manquée avant et après HPSS | 21 | 3 | 0 |
| Faux positif Pop sous HPSS | 41 | 3 | 0 |
| Pop manquée par les trois ViT | 71 | 3 | 0 |

- Ouvrir les fiches d’écoute (`data/interim/error-review-20261021T233320701672Z/index.html`, non versionné)
- Sélection et provenance (`data/interim/error-review-20261021T233320701672Z/queue.json`, non versionné)
- Modèle d’annotations vierge (`data/interim/error-review-20261021T233320701672Z/annotations_template.json`, non versionné)

Sélection déterministe par empreinte avec graine fixe, catégories rares d’abord et au plus un morceau par artiste. Les cas sont ciblés pour comprendre des phénomènes ; ils ne constituent pas un échantillon représentatif des taux d’erreur. La catégorie « manquée par les trois ViT » n’exige pas qu’ils prédisent tous le même mauvais genre.

## Interprétation et suite

La faible F1 Pop combine des faux négatifs et des faux positifs. Les confusions et désaccords localisent les erreurs, mais ne démontrent ni une mauvaise annotation ni une cause acoustique. Les probabilités maximales ne sont pas calibrées. Les genres FMA détaillés et la langue déclarée sont des indices de revue, pas des vérités auditives.

Écouter avant d’afficher les prédictions ; décrire voix, rythme, timbres, évolution de l’extrait et ambiguïtés. Exporter les observations JSON. Aucune modification automatique des labels, partitions ou originaux. Une éventuelle nouvelle expérience sera motivée par des observations documentées ; bloc 6 + HPSS reste non testé.

## Reproduction

Commande : `.\.venv\Scripts\python.exe scripts/analyze_validation_errors.py`.

Chaque exécution crée un nouveau dossier d’écoute et conserve les précédents. Les annotations exportées doivent être conservées séparément ; la synthèse automatisée ne les marque jamais comme réalisées.

[Résultats détaillés, toutes les paires et empreintes](VALIDATION_ERRORS.json).
