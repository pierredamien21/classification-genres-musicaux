# Diagnostic ciblé : résultats et interprétation

Mis à jour : 26 octobre 2026. Diagnostic terminé sur les 19 morceaux de la revue,
soit 57 fenêtres de validation. Aucun entraînement ni changement d’étiquette.

## Ce qui a été vérifié

Les MP3 originaux ont été redécodés, puis les fenêtres, log-Mel natifs, embeddings
des trois ViT et probabilités agrégées ont été recalculés. Les empreintes des
sources, versions, fichiers audio, caches et modèles sont contrôlées. Les seuils
étaient fixés dans [la configuration](../configs/audio_diagnostic.json) avant
l’exécution : 0,0001 dB, 0,0001 pour les embeddings et 0,00001 pour les probabilités.

| Mesure sur les 57 fenêtres | Résultat |
|---|---:|
| Écart maximal avec le cache log-Mel | 0 dB |
| Écart maximal avec les embeddings enregistrés, trois ViT | 0 |
| Écart maximal des probabilités par morceau | 1,94 × 10⁻¹⁶ |
| Fenêtres contenant du padding | 11/57 |
| Padding maximal dans une fenêtre de 10 s | 23,375 ms |
| RMS sur les échantillons valides | −23,24 à −7,04 dBFS |

Les fractions maximales de cases spectrales au-dessus de 20 dB sont de 2,44 %,
1,67 % et 0,34 % pour les canaux original, harmonique et percussif. Sous −80 dB,
elles atteignent respectivement 0,062 %, 1,08 % et 2,04 %. Ces valeurs décrivent
le bornage de la représentation ; elles ne démontrent pas une saturation audible.
Le RMS global d’une fenêtre n’exclut pas des passages brefs de faible amplitude.

## Une piste temporelle, encore à confirmer

Avec HPSS, les trois fenêtres donnent des genres différents pour 16 des 19
morceaux. Parmi les 14 morceaux mal classés après moyenne des probabilités,
7 ont au moins une fenêtre correctement classée selon l’étiquette FMA.
Par exemple, `120201` est classé Pop / Hip-Hop / Pop selon la fenêtre, mais
Hip-Hop après agrégation : la moyenne des probabilités n’est pas un vote majoritaire.

Ce constat motive une analyse descriptive sur les **780 morceaux de validation**,
par genre et par modèle, avant de modifier l’agrégation. Cette analyse complète
n’est pas encore réalisée. Les 19 cas sont une sélection ciblée : leurs proportions
d’erreurs ne décrivent pas les performances globales. Choisir une fenêtre parce
qu’on connaît sa vraie étiquette serait une fuite d’information.

## Portée et limites

Aucun écart dépassant les tolérances n’a été trouvé avec le pipeline enregistré.
Cela ne prouve ni que les choix de prétraitement sont optimaux, ni qu’un défaut
commun à l’extraction et au recalcul est absent. Les CNN n’ont pas été réexécutés ;
leur entrée log-Mel commune a été vérifiée. Le jeu de test n’a pas été évalué.

Sept figures ont été générées ; celles de `064630` et `120201` ont été inspectées
visuellement. Ce contrôle ne constitue pas une écoute musicale. L’utilisateur a
confirmé l’affichage de la page et le son du premier extrait ; aucune annotation
musicale n’a été produite. Export des annotations et persistance restent non vérifiés.

## Preuves et reproduction

- [Rapport par morceau et figures](../reports/DIAGNOSTIC_AUDIO.md).
- [Mesures, prédictions et empreintes](../reports/AUDIO_DIAGNOSTIC.json).
- [Pointeur avec empreinte du rapport](../reports/latest_audio_diagnostic.json).
- [Contrôles unitaires](../tests/test_audio_diagnostic.py) : cinq cas nouveaux ;
  suite complète de 99 tests réussie, Ruff réussi.

Depuis la racine du projet, exécuter `.venv/Scripts/python.exe scripts/diagnose_review_audio.py`.
La commande vérifie les caches existants puis recalcule la sélection. Elle écrit
un nouveau dossier de figures et actualise les rapports du diagnostic.
