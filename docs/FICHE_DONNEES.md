# Fiche des données

Mis à jour : 11 novembre 2026. Données de la V2 ; la V1 utilisait le seul ensemble de développement ci-dessous, découpé autrement.

## Source

[Free Music Archive (FMA)](https://github.com/mdeff/fma) :
- `fma_metadata` : métadonnées des morceaux ;
- `fma_small` : 8 000 extraits de 30 s ;
- `fma_large` : extraits de 30 s.

Seuls les morceaux nécessaires ont été lus dans `fma_large.zip`, par requêtes HTTPS partielles : environ 1 Go au lieu de 93 Go. Taille, CRC-32 et SHA-256 de chaque fichier sont consignés.

**Rien de l'audio n'est versionné ni publiable.** `data/raw/` est exclu de Git. Chaque morceau a sa propre licence Creative Commons (champ `track.license` de `tracks.csv`) :
- 82 % des morceaux du développement et 87 % des morceaux V2 sont **non commerciaux** ;
- 35 % et 46 % interdisent les modifications.

Le dépôt ne contient que des identifiants, des empreintes, des prédictions et des statistiques.

## Classes

Les 8 genres de FMA-small, dans l'ordre de `configs/labels.json` : Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock. L'étiquette est le `genre_top` de FMA ; aucune n'a été modifiée.

Plusieurs classes ne sont pas des genres acoustiques : Experimental, Instrumental, International.

## Cohortes

| Cohorte | Morceaux | Artistes | Origine | Usage |
|---|---:|---:|---|---|
| Développement | 7 857 | 2 296 | FMA-small (ex-train, validation et test V1) | 5 plis groupés par artiste, stratifiés par genre ; tout le choix de modèles |
| Test V2 | 501 | 501 | FMA-medium/large, artistes absents de FMA-small | Une seule inférence, après le verrou 3 |
| Hors domaine | 340 | 340 | 8 genres FMA hors des 8 classes | Rejet « genre inconnu » |
| Sonde Gospel | 12 | 12 | Gospel FMA | Rapportée à part |

- **Plis** : 1 569 à 1 579 morceaux chacun ([`reports/v2/dev_folds.csv`](../reports/v2/dev_folds.csv)).
- **Test V2** : 64 morceaux par genre, sauf International (53, faute d'artiste inutilisé), un morceau par artiste.
- **Hors domaine** : Blues 27, Classical 64, Country 36, Easy Listening 4, Jazz 64, Old-Time / Historic 64, Soul-RnB 17, Spoken 64.
- Listes et empreintes : [`reports/v2/final_cohorts.csv`](../reports/v2/final_cohorts.csv).

## Sélection et audit (avant tout modèle)

- **Règles de tirage** ([protocole](PROTOCOLE_V2.md)) : rangs pseudo-aléatoires à graine fixe, un morceau par artiste, artistes absents du développement.
- **Audit** des 1 092 fichiers téléchargés :
  - décodage, durée, silence et niveau ;
  - doublons PCM exacts et quasi-doublons Chromaprint, contre le développement et entre cohortes.
- **Bilan** : 75 échecs remplacés par le suivant du même rôle et du même genre, soit 50 remplaçants utilisés. Aucune écoute ni aucun réétiquetage.
- **Amendement 1** : 45 morceaux Pop de `fma_large` sont des extraits de 0 à 15 s. Des remplaçants Pop d'au moins 30 s ont été ajoutés, avant le verrou 2.
- **Amendement 2** : 20 morceaux retenus produisent des messages d'erreur du décodeur. Ils sont conservés, avec une analyse de sensibilité sans eux.
- Preuves : [audit](../reports/v2/audit.csv), [verrou 2](../reports/v2/lock2.json).

## Écarts connus entre développement et test

- **Popularité** : le test est beaucoup moins écouté (médiane de 374 écoutes contre 2 493).
- **Artistes** : ils sont moins prolifiques dans FMA.
- **Points communs** : débit, nombre de genres FMA et années proches.

Ces écarts n'expliquent pas la baisse de score ([interprétation](INTERPRETATION_V2.md), section 3).

## Limites

- Étiquettes FMA non vérifiées ; l'écoute de 18 extraits en suggère quelques-unes de discutables.
- Le développement est déséquilibré par artiste : 3,4 morceaux par artiste en moyenne.
- International est sous-représenté dans le test.
- Le recouvrement entre les données de préentraînement de MAEST (Discogs20, privé) et FMA est invérifiable.
