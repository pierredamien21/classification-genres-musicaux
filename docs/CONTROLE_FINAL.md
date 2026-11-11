# Contrôle final : chaque critique et son traitement

Mis à jour : 11 novembre 2026. Ce tableau reprend chaque point de la [critique du 8 novembre](archive/CRITIQUE_PROJET.md) et indique son traitement par la V2.

Statuts :
- **Corrigé** : le défaut n'existe plus.
- **Mesuré** : la question a reçu une réponse chiffrée.
- **Limite assumée** : le défaut reste, documenté.

## Critiques scientifiques

| # | Critique | Statut | Ce qui a été fait, preuve |
|---|---|---|---|
| 1 | La question « ViT » n'est pas testée : pas d'affinage, pas de modèle audio préentraîné | **Mesuré** | AST, PANNs et MAEST en sonde linéaire, plus un ViT natif ; ViT affiné sur le pli 0. MAEST bat le meilleur système V1 de +0,091 sur le test V2 (H2 confirmée). [Rapport](RAPPORT_V2.md) |
| 2 | L'évaluation repose sur une poignée d'artistes | **Corrigé** | Test V2 de 501 artistes, un morceau chacun ; validation croisée groupée par artiste ; métriques pondérées par artiste. L'écart V1 vient de la composition par artiste. [Interprétation](INTERPRETATION_V2.md) |
| 3 | Les intervalles bootstrap de la macro-F1 sont décalés | **Corrigé** | Intervalles BCa, validés contre SciPy, avec un recomptage indépendant concordant. `src/music_genre/evaluation/final_v2.py` |
| 4 | Le CNN de comparaison est trop faible | **Corrigé en partie** | CNN renforcé : normalisation de batch, SpecAugment, patience 10, 3 graines. Il reste loin derrière (0,44–0,46 sur le test). Limites : 24 000 paramètres, 80 % du développement. [CNN](../reports/v2/cnn_v2_seeds.csv) |
| 5 | Le passage spectrogramme → image n'est pas testé | **Mesuré** | ViT à résolution native : 0,569 contre 0,553 en CV. Bruit, gain et MP3 mesurés sur tout le développement, augmentation testée. [ViT natif](../reports/v2/cv_results.csv), [phase 4](../reports/v2/phase4.json) |
| 6 | Capacité limitée, grille de C trop étroite | **Mesuré** | Grille étendue jusqu'à 1e-5. Une tête MLP fait moins bien (0,668 contre 0,684). [Tableau CV](../reports/v2/cv_results.csv) |
| 7 | Taxonomie et couverture : Gospel, genres inconnus | **Mesuré, puis limite assumée** | 340 morceaux hors domaine et une sonde Gospel. Le rejet est inefficace (AUROC 0,67, 89 % acceptés). Les classes non acoustiques restent ; l'écoute en montre les effets. |

## Critiques d'ingénierie

| Critique | Statut | Ce qui a été fait, preuve |
|---|---|---|
| Pas de dépôt Git | **Corrigé** | Dépôt depuis le 8 novembre, étiquette `v1-final`, hooks Ruff et tests |
| Sur-ingénierie, scripts dupliqués | **Corrigé en partie** | Scripts V1 ponctuels retirés de `main` : 36 scripts et 4 838 lignes, contre environ 9 500. Extraction unifiée (`extract_features.py --config`). Les sous-classes V1 épinglées restent. |
| Documentation trop abondante | **Corrigé** | Un jeu réduit : README, fiches des données et du modèle, rapport V2, architecture, reproduction, index. 26 documents V1 sont dans `docs/archive/`. 10 restent en place, car ils sont épinglés ou cités par des preuves V1. |
| La CLI n'a que `doctor` | **Corrigé** | `python -m music_genre predict` existe (exports V1) |
| La démo décode l'audio deux fois | **Limite assumée** | Toujours vrai (`app/main.py`). Effet mineur, en local seulement. |
| Environnement émulé, réinstallation non testée | **Corrigé en partie** | Réinstallation depuis un clone propre et un cache vide sur ce poste : 76 paquets, 241 tests réussis, 68 ignorés pour absence de données. Restent non testés : un autre poste ou compte, et Python ARM64 natif. [Reproduction](REPRODUCTION.md) |

## Recommandations de la critique

| Recommandation | Suite donnée |
|---|---|
| 1. `git init` | Fait |
| 2. Modèle audio préentraîné et nouvelle évaluation indépendante | Fait : MAEST, AST, PANNs ; test V2 sur des artistes absents de FMA-small |
| 3. Analyse par artiste, pondération, BCa | Fait |
| 4. Renforcer le CNN | Fait, mais il reste une simple référence |
| 5. Résolution native, augmentation | Fait. L'augmentation, efficace contre le bruit, n'a pas été adoptée, en application de la règle préenregistrée. |
| 6. Fusionner les scripts, réduire la documentation | Fait en partie (voir plus haut) |

## Ce qui reste à l'auteur

- Choisir une licence pour le code avant la publication. Le dépôt n'en a aucune.
- Publier le dépôt : code et rapports, jamais l'audio FMA, ni les poids, ni `data/interim/`.
- Tourner la vidéo de démonstration.
- Tester la réinstallation sur un autre poste ou un autre compte.
