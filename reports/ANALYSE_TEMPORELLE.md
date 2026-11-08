# Désaccords temporels sur toute la validation

780 morceaux, 2 340 fenêtres par modèle, trois ViT avec classificateurs exportés inchangés. Aucun entraînement, nouvelle extraction audio ou accès aux données de test.

## Désaccords et agrégation

| Modèle | Désaccord entre fenêtres | Erreurs de la moyenne | Erreurs avec ≥1 fenêtre correcte | Moyenne correcte sans fenêtre correcte | Majorité unique différente de la moyenne | Sans majorité unique |
|---|---:|---:|---:|---:|---:|---:|
| ViT bloc 12 original | 429/780 | 348 | 105 | 1 | 52 | 73 |
| ViT bloc 6 original | 346/780 | 321 | 80 | 1 | 33 | 64 |
| ViT bloc 12 HPSS | 392/780 | 314 | 73 | 1 | 44 | 89 |

Un désaccord signifie que les trois argmax ne sont pas identiques. Une majorité unique exige au moins deux votes identiques ; trois genres distincts n’en ont pas. Les égalités d’argmax suivent l’ordre des classes. Aucun score d’un nouveau vote majoritaire n’est sélectionné.

![Comptages par nombre de fenêtres correctes](../reports/figures/temporal-20261025T232823735697Z/temporal_counts.png)

## Même classificateur, passages différents

| Modèle | Entrée évaluée | Macro-F1 | Accuracy |
|---|---|---:|---:|
| ViT bloc 12 original | Moyenne des trois | 0.5447 | 0.5538 |
| ViT bloc 12 original | 0–10 s | 0.4982 | 0.5038 |
| ViT bloc 12 original | 10–20 s | 0.4896 | 0.4962 |
| ViT bloc 12 original | 20–30 s | 0.5370 | 0.5410 |
| ViT bloc 6 original | Moyenne des trois | 0.5780 | 0.5885 |
| ViT bloc 6 original | 0–10 s | 0.5429 | 0.5538 |
| ViT bloc 6 original | 10–20 s | 0.5397 | 0.5449 |
| ViT bloc 6 original | 20–30 s | 0.5544 | 0.5641 |
| ViT bloc 12 HPSS | Moyenne des trois | 0.5844 | 0.5974 |
| ViT bloc 12 HPSS | 0–10 s | 0.5323 | 0.5436 |
| ViT bloc 12 HPSS | 10–20 s | 0.5129 | 0.5256 |
| ViT bloc 12 HPSS | 20–30 s | 0.5330 | 0.5410 |

Ces évaluations par position conservent les modèles appris sur trois fenêtres. Elles ne sont ni des réentraînements sur une seule fenêtre ni une sélection de la meilleure position.

## Détail par genre

| Modèle | Genre FMA | Morceaux | Artistes | Désaccords | Erreurs moyenne | Erreurs avec fenêtre correcte | Moyenne correcte sans fenêtre correcte |
|---|---|---:|---:|---:|---:|---:|---:|
| ViT bloc 12 original | Electronic | 99 | 36 | 52 | 37 | 12 | 0 |
| ViT bloc 12 original | Experimental | 100 | 45 | 42 | 52 | 10 | 0 |
| ViT bloc 12 original | Folk | 99 | 25 | 51 | 36 | 9 | 0 |
| ViT bloc 12 original | Hip-Hop | 100 | 17 | 45 | 24 | 9 | 0 |
| ViT bloc 12 original | Instrumental | 93 | 11 | 58 | 48 | 21 | 0 |
| ViT bloc 12 original | International | 98 | 24 | 53 | 34 | 15 | 0 |
| ViT bloc 12 original | Pop | 94 | 26 | 70 | 85 | 18 | 0 |
| ViT bloc 12 original | Rock | 97 | 43 | 58 | 32 | 11 | 1 |
| ViT bloc 6 original | Electronic | 99 | 36 | 39 | 33 | 8 | 1 |
| ViT bloc 6 original | Experimental | 100 | 45 | 46 | 50 | 13 | 0 |
| ViT bloc 6 original | Folk | 99 | 25 | 44 | 30 | 11 | 0 |
| ViT bloc 6 original | Hip-Hop | 100 | 17 | 29 | 22 | 5 | 0 |
| ViT bloc 6 original | Instrumental | 93 | 11 | 45 | 43 | 10 | 0 |
| ViT bloc 6 original | International | 98 | 24 | 37 | 28 | 9 | 0 |
| ViT bloc 6 original | Pop | 94 | 26 | 60 | 82 | 11 | 0 |
| ViT bloc 6 original | Rock | 97 | 43 | 46 | 33 | 13 | 0 |
| ViT bloc 12 HPSS | Electronic | 99 | 36 | 54 | 35 | 10 | 0 |
| ViT bloc 12 HPSS | Experimental | 100 | 45 | 43 | 47 | 13 | 0 |
| ViT bloc 12 HPSS | Folk | 99 | 25 | 42 | 26 | 5 | 0 |
| ViT bloc 12 HPSS | Hip-Hop | 100 | 17 | 44 | 29 | 9 | 0 |
| ViT bloc 12 HPSS | Instrumental | 93 | 11 | 46 | 39 | 9 | 0 |
| ViT bloc 12 HPSS | International | 98 | 24 | 48 | 27 | 8 | 0 |
| ViT bloc 12 HPSS | Pop | 94 | 26 | 64 | 82 | 10 | 1 |
| ViT bloc 12 HPSS | Rock | 97 | 43 | 51 | 29 | 9 | 0 |

## Contrôles et limites

Les trois caches, modèles, prédictions enregistrées, sources d’extraction et versions sont vérifiés. Chaque morceau correspond exactement au manifeste de validation, avec les fenêtres 0/10/20 s, le même artiste, label, fichier source et minutage entre caches. Les moyennes reproduisent les probabilités et métriques historiques.

Écart maximal des probabilités : 2.22e-16. Durée totale du diagnostic : 2.27 s.

La présence d’une fenêtre correcte utilise la vérité terrain : c’est une description a posteriori, pas un sélecteur utilisable à l’inférence ni un gain atteignable démontré. Des désaccords ne prouvent pas un changement réel de genre, un défaut de label ou un problème de prétraitement. Les morceaux d’un artiste ne sont pas indépendants ; aucun intervalle de confiance ni test de significativité n’est revendiqué. La validation reste un jeu de développement. Aucune écoute subjective n’a été réalisée.

[Protocole et interprétation](../docs/ANALYSE_TEMPORELLE.md) · [Rapport numérique et probabilités](TEMPORAL_ANALYSIS.json)

Commande : `.venv/Scripts/python.exe scripts/analyze_temporal_validation.py`.
