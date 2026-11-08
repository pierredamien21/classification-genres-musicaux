# Audit technique de FMA-small

Exécuté le 2026-10-20 ; durée 6.4 minutes.

**8,000 morceaux contrôlés** : 7815 sans alerte selon ces contrôles, 179 à examiner, 6 exclus pour défaut bloquant.

Les fichiers originaux et les partitions officielles sont conservés. « Sans alerte » ne garantit ni l’absence de défaut audible ni l’absence de doublon perceptuel.

- [Manifeste complet](../data/manifests/audit-20261020T075240326780Z/manifest.csv)
- [File de revue, exclusions comprises](../data/manifests/audit-20261020T075240326780Z/review_queue.csv)
- [Provenance et résultats détaillés](../data/manifests/audit-20261020T075240326780Z/report.json)

SHA-256 du manifeste : `97afd9f768020d76c451e0a96e71b6c4270122105162ad699b3405d01324ee95`.

## Contrôles et interprétation

Lecture intégrale par libsndfile, blocs d’une seconde, fréquence et canaux d’origine, sans rééchantillonnage. Empreinte du fichier et empreinte du PCM float32 little-endian incluant fréquence et nombre de canaux. Diagnostics natifs capturés par fichier.

Seuils fixés avant l’audit dans `configs/audit.json` : durée hors de 29–31 s signalée ; blocs RMS ≤ −60 dBFS considérés silencieux et revue à partir de 50 % de durée silencieuse ; revue si au moins 1 % des échantillons ont une amplitude absolue ≥ 0,999. Cette dernière mesure indique une proximité de la pleine échelle, pas une preuve de saturation.

Exclusion technique uniquement si décodage impossible, signal vide/non fini ou entièrement nul. Les autres alertes, y compris les doublons, restent en revue. Aucun genre, score de modèle ou résultat de test n’intervient dans ces critères.

## Répartition des statuts

| Partition | Sans alerte | À examiner | Exclus |
|---|---:|---:|---:|
| train | 6264 | 130 | 6 |
| validation | 778 | 22 | 0 |
| test | 773 | 27 | 0 |

| Genre | Sans alerte | À examiner | Exclus |
|---|---:|---:|---:|
| Electronic | 977 | 22 | 1 |
| Experimental | 951 | 48 | 1 |
| Folk | 992 | 8 | 0 |
| Hip-Hop | 973 | 24 | 3 |
| Instrumental | 980 | 20 | 0 |
| International | 995 | 5 | 0 |
| Pop | 975 | 25 | 0 |
| Rock | 972 | 27 | 1 |

## Alertes

Un morceau peut cumuler plusieurs alertes.

| Alerte | Morceaux |
|---|---:|
| decoder_diagnostic | 58 |
| duplicate_pcm_sha256 | 24 |
| missing_license | 5 |
| mostly_silent | 5 |
| near_full_scale | 96 |

Durées décodées : minimum 29.965 s, médiane 29.977 s, maximum 30.003 s. La durée des métadonnées peut décrire le morceau d’origine et ne remplace pas celle de l’extrait.

## Échecs de décodage

| Morceau | Motif |
|---|---|
| 098565 | decode_failed |
| 098567 | decode_failed |
| 098569 | decode_failed |
| 099134 | decode_failed |
| 108925 | decode_failed |
| 133297 | decode_failed |

Les 6 échecs ont été reproduits en lecture séquentielle. Les fichiers existent et leurs octets correspondent à l’archive d’origine. Le message « File does not exist » de libsndfile ne décrit donc pas ici une absence réelle de fichier. Cela ne prouve pas qu’aucun autre décodeur puisse récupérer un signal.
[Vérification détaillée](../data/manifests/audit-20261020T075240326780Z/failure_recheck.json).

Morceaux du premier benchmark avec diagnostic natif : 054568.

## Doublons et fuite entre partitions

- `source_sha256` : 0 groupe(s), dont 0 entre partitions.
- `pcm_sha256` : 12 groupe(s), dont 0 entre partitions.
  - Morceaux 144933, 145702 ; partitions train.
  - Morceaux 114223, 115290 ; partitions test.
  - Morceaux 114236, 125820 ; partitions train.
  - Morceaux 114222, 115292 ; partitions test.
  - Morceaux 114235, 125821 ; partitions train.
  - Morceaux 114212, 115294 ; partitions test.
  - Morceaux 144938, 145707 ; partitions train.
  - Morceaux 120470, 133016 ; partitions train.
  - Morceaux 130170, 130650 ; partitions train.
  - Morceaux 045102, 051785 ; partitions train.
  - Morceaux 074348, 086761 ; partitions train.
  - Morceaux 057371, 072149 ; partitions train.

Identifiants d’artiste partagés entre partitions : 0 (somme des intersections deux à deux).

Les empreintes exactes ne détectent pas toutes les reprises, versions réencodées, changements de volume ou extraits décalés. Une recherche perceptuelle reste à mener avant de déclarer l’absence de fuite de contenu.

## Licences et publication

La licence de chaque morceau est reprise des métadonnées dans le manifeste. Ce relevé ne vaut pas validation des droits de redistribution ; ne pas publier les fichiers audio avec le portfolio.

## Suite

Examiner la file de revue, comparer au besoin les fichiers problématiques avec un second décodeur, puis consigner les décisions dans un journal distinct. Construire ensuite un manifeste d’entraînement approuvé, sans modifier les originaux ni déplacer un morceau vers une autre partition. Les alertes de durée ou de niveau ne justifient pas à elles seules une suppression.
