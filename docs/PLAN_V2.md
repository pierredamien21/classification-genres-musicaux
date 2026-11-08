# Plan V2 — combler les lacunes identifiées

8 novembre 2026. État : **prévu, rien n'est lancé.** Ce plan répond point par point à
[CRITIQUE_PROJET.md](CRITIQUE_PROJET.md). La V1 reste figée : elle n'est ni modifiée
ni réévaluée comme si son test était encore neuf.

## Objectif réaliste

« Parfait sans aucune lacune » n'existe pas en apprentissage automatique. L'objectif
vérifiable est : **chaque lacune connue est soit corrigée, soit mesurée et documentée,
et aucune conclusion ne dépasse ce que les preuves permettent.** Le tableau final de
la phase 7 sert de liste de contrôle.

## Faits vérifiés qui structurent le plan

- **Le test V1 est consommé.** Toute nouvelle sélection a besoin d'un nouveau test.
- **FMA-medium ne suffit pas** pour un test équilibré d'artistes jamais vus :
  International 1 morceau, Pop 22, Folk 78.
- **FMA-large suffit.** Morceaux de medium/large, des 8 genres, d'artistes absents
  de FMA-small (calcul sur `tracks.csv`) :

  | Genre | Artistes jamais vus | Genre | Artistes jamais vus |
  |---|---:|---|---:|
  | Electronic | 1 628 | Instrumental | 287 |
  | Experimental | 2 103 | International | **67** |
  | Folk | 130 | Pop | 186 |
  | Hip-Hop | 116 | Rock | 1 578 |

- Des genres hors des 8 classes existent pour tester le rejet : Classical 546,
  Old-Time 509, Jazz 359, Country 161, Soul-RnB 121, Spoken 107, Blues 56 (medium seul).
- Gospel n'a que 66 morceaux dans FMA : impossible d'en faire une classe fiable ;
  il peut servir de sonde hors domaine. Rap (638) est un sous-genre de Hip-Hop.
- Disque E: : 434 Go libres ; `fma_large.zip` fait environ 93 Go.
- Un Python 3.14 ARM64 natif est installé sur le poste ; la compatibilité de
  PyTorch et librosa en ARM64 natif reste **à vérifier**.
- Windows 11 Famille : Windows Sandbox n'est pas disponible pour tester une
  réinstallation propre ; il faudra un autre compte, un autre poste ou une VM.

## Phase 0 — Fondations (≈ 1 jour, aucun calcul)

1. `git init`, premier commit des sources, configurations, docs et rapports légers ;
   étiquette `v1-final`. Les données et les poids restent hors dépôt.
2. Corriger les incohérences relevées : nombre de tests du README, fiche du modèle
   (« interface non livrée »), date et tableau de l'index, `__init__.py`
   « à implémenter ».
3. Hooks locaux : `pytest` et `ruff` avant chaque commit.

**Sortie :** dépôt propre, 271 tests verts, aucune contradiction documentaire.

## Phase 1 — Simplifier le code sans changer la V1 (≈ 3–4 jours)

1. Un seul extracteur paramétré (représentation, canaux, bloc, pooling) à la place
   des quatre `extract_vit*.py` et des sous-classes qui tronquent le modèle.
2. Un seul script d'entraînement CNN (base, extension, session utilisateur).
3. Un pipeline unique : configuration → extraction → apprentissage → évaluation.
4. Commande `music-genre predict fichier.mp3`, qui utilise la même fonction que la démo.
5. Démo : décodage unique par requête, suppression des fichiers envoyés après analyse.
6. Environnement : essai ARM64 natif mesuré par le benchmark existant ;
   dépendances verrouillées avec empreintes.

**Preuve exigée :** les 9 exports V1 reproduisent leurs probabilités de validation à
10⁻⁶ près avec le nouveau code. Sinon on n'avance pas.

## Phase 2 — Protocole V2, verrouillé avant toute expérience (≈ 2 jours + téléchargement)

1. **Téléchargement de FMA-large (~93 Go), à valider par toi avant lancement.**
2. **Test V2** : tiré au hasard (graine fixée) parmi les artistes absents de FMA-small,
   **un morceau par artiste**, 64 artistes par genre, soit 512 morceaux et 512 artistes
   (International limite à 67). Audit technique seulement, sans écoute ni regard sur
   les prédictions. Taille finale fixée par simulation de puissance à partir de la
   variance des différences appariées V1.
3. **Ensemble hors domaine** : environ 64 morceaux par genre absent (Classical, Jazz,
   Old-Time, Country, Soul-RnB, Spoken, Blues), plus Gospel en sonde, artistes disjoints.
4. **Développement** : les 7 857 morceaux FMA-small (train, validation et test V1)
   avec validation croisée en 5 plis groupés par artiste pour les modèles linéaires ;
   pour les modèles entraînés (CNN, fine-tuning), un découpage fixe par artiste afin
   de tenir le budget CPU.
5. **Hypothèses déclarées à l'avance**, en nombre limité, avec correction de Holm :
   - H1 : réplication de la V1 figée sur le test V2 (le candidat V1 bat-il encore le ViT original ?) ;
   - H2 : meilleur modèle audio préentraîné contre candidat V1 ;
   - H3 : meilleur modèle V2 contre CNN renforcé.
6. **Statistiques** : intervalles BCa par artiste, macro-F1 en métrique principale,
   accuracy et F1 par genre, calibration (ECE, Brier), rejet (AUROC, FPR à 95 %).
7. Verrouillage par empreintes, comme T6 en V1.

**Sortie :** manifeste V2 verrouillé ; aucun modèle V2 encore entraîné.

## Phase 3 — Comparateurs manquants (≈ 1–2 semaines, surtout du calcul CPU)

Chaque modèle commence par un benchmark sur 64 fenêtres pour estimer son coût.

1. **Modèles audio préentraînés gelés + sonde linéaire** : AST (AudioSet),
   PANNs CNN14, MAEST (Discogs). Exposition au préentraînement documentée.
   MusicFM est exclu (préentraîné sur FMA-large, qui contient le test V2).
2. **ViT image amélioré** : entrée à résolution native avec interpolation des
   positions (méthode AST), contre le redimensionnement 224 × 224 actuel.
3. **Tête non linéaire** (petit MLP) sur les meilleures représentations gelées,
   puisque la V1 sous-apprend (F1 train 0,635).
4. **Fine-tuning partiel** du ViT-Tiny (derniers blocs), seulement si le benchmark
   le rend faisable sur CPU.
5. **CNN renforcé** : normalisation de batch, SpecAugment, patience 10, trois graines.
6. Grille de C élargie (10⁻³ à 10).

**Sortie :** tous les candidats comparés en validation croisée ; un seul candidat V2
choisi selon la règle déclarée en phase 2.

## Phase 4 — Robustesse et calibration (≈ 3–4 jours)

1. Augmentation par bruit, gain et recompression MP3 à l'entraînement des têtes et du CNN.
2. Robustesse mesurée sur tout le développement, pas sur 64 morceaux :
   SNR 30/20/10 dB, ±6 dB de gain, MP3 64 kb/s.
3. Calibration par température sur les plis de validation.
4. Seuil de rejet « genre inconnu » réglé sans toucher l'ensemble hors domaine du test.

## Phase 5 — Évaluation finale V2, une seule fois (≈ 1 jour)

Inférence unique sur le test V2 et l'ensemble hors domaine, recomptage indépendant,
rapport des trois hypothèses, y compris les résultats défavorables.

## Phase 6 — Interprétation (≈ 2–3 jours, en partie toi)

1. Analyse par artiste et par genre, et explication de l'écart V1 validation/test.
2. **Toi seul** : écoute annotée des erreurs avec la fiche existante.
3. Comparaison V1/V2 présentée comme une réplication, pas comme un gain sur le test V1.

## Phase 7 — Livraison (≈ 2–3 jours, en partie toi)

1. Documentation réduite à : README, fiche des données, fiche du modèle, rapport V2,
   architecture, reproduction, index. L'historique V1 va dans `docs/archive/`,
   sans suppression.
2. Réinstallation testée depuis zéro sur un autre compte ou un autre poste.
3. **Toi seul** : publication du dépôt (code et rapports, jamais l'audio FMA ;
   licences des poids vérifiées) et vidéo de démonstration.
4. Tableau final de contrôle : chaque point de la critique → corrigé, mesuré ou limite assumée.

## Ce qui dépend de toi

- Accord pour télécharger environ 93 Go (phase 2).
- Accepter plusieurs heures de calcul CPU par modèle (phase 3), ou prévoir un GPU distant.
- Écoute annotée, publication, vidéo et test sur un autre poste.

## Risques connus

- Les étiquettes de FMA-large peuvent être plus bruitées que celles de FMA-small :
  c'est un changement de distribution à mesurer et à documenter, pas à corriger
  après avoir vu les résultats.
- Coût CPU d'AST et du fine-tuning encore inconnu : le benchmark décide.
- Même avec 512 artistes, l'intervalle d'une accuracy reste d'environ ±4 points ;
  les différences appariées seront plus précises.
