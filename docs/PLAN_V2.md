# Plan V2 — combler les lacunes identifiées

8 novembre 2026. État : **prévu, rien n'est lancé.** Ce plan répond point par point à
[CRITIQUE_PROJET.md](archive/CRITIQUE_PROJET.md). La V1 reste figée : elle n'est ni modifiée
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
2. Corriger les incohérences relevées dans les documents **non épinglés** : README
   (nombre de tests, principe CLI), date, état et tableau de l'index.
3. Hook versionné : `pytest` et `ruff` (sur `src`, `tests`, `scripts`, `app`) avant
   chaque commit.

**Sortie :** dépôt propre, 271 tests verts, aucune contradiction documentaire.

**Réalisé le 8 novembre.** Laissés
volontairement intacts, car épinglés par empreinte dans les preuves V1 : la fiche du
modèle, les `__init__.py` (docstrings « à implémenter ») et les notebooks (16
avertissements Ruff). Ils changeront avec la V2 verrouillée.

## Phase 1 — Simplifier le code sans changer la V1 (≈ 3–4 jours)

1. Un seul extracteur paramétré (représentation, canaux, bloc, pooling) à la place
   des quatre `extract_vit*.py` et des sous-classes qui tronquent le modèle.
2. Un seul script d'entraînement CNN (base, extension, session utilisateur).
3. Un pipeline unique : configuration → extraction → apprentissage → évaluation.
4. Commande `music-genre predict fichier.mp3`, qui utilise la même fonction que la démo.
5. Démo : décodage unique par requête (la purge des fichiers envoyés existe déjà).
6. Docstrings des `__init__.py` à jour, dans le même changement que la refonte.
7. Environnement : essai ARM64 natif mesuré par le benchmark existant ;
   dépendances verrouillées avec empreintes.

**Preuve exigée :** les 9 exports V1 reproduisent leurs probabilités de validation à
10⁻⁶ près avec le nouveau code. Sinon on n'avance pas. Les sources V1 épinglées
restent consultables via l'étiquette `v1-final` ; les contrôles V1 se rejouent sur
cette étiquette (par exemple dans un `git worktree`), pas sur le code refondu.

**Réalisé le 8 novembre** ([équivalence V1](../reports/V1_EQUIVALENCE.json)) : reproduction exacte
(écart nul) des 9 exports, des 5 représentations et de l'entraînement CNN. Restent
ouverts : l'essai ARM64 natif et le verrouillage des dépendances par empreintes, qui
demandent des téléchargements. Abandonné : le décodage unique de la démo.

## Phase 2 — Protocole V2, verrouillé avant toute expérience (≈ 2 jours + téléchargement)

1. **Téléchargement de FMA-large (~93 Go), validé par l'auteur avant lancement.**
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

**Réalisée le 8 novembre** — [protocole préenregistré](PROTOCOLE_V2.md), [verrou 2](../reports/v2/lock2.json).
Ajustements : hors domaine plafonné par les artistes disponibles (340 et non 8 × 64) ;
H1 seule primaire (puissance environ 56 %) ; téléchargement partiel par lectures HTTP
`Range` (≈ 1 Go au lieu de 93 Go) ; amendement 1 (remplaçants Pop) avant le verrou 2.
Test final : 501 morceaux (International 53), 340 hors domaine, 12 Gospel.

## Phase 3 — Comparateurs manquants (≈ 1–2 semaines, surtout du calcul CPU)

**État (10 novembre 2026) : terminée.** Points 1 à 6 réalisés ; grille de C élargie
à 10⁻⁵–10, optimum intérieur (10⁻⁴ ou 10⁻³) ; MAEST bloc 12 en tête (0,684 en CV
contre 0,553 pour la V1). Tête MLP : 0,668 sur MAEST et 0,650 sur AST, sous la sonde
linéaire. CNN renforcé : 0,618 ± 0,015 sur le pli 0 (trois graines), contre 0,728 pour
MAEST sur ce pli. ViT natif (sans redimensionnement) : bloc 6 à 0,569 contre 0,553.
Fine-tuning partiel (blocs 5–6, une graine) : 0,661 sur le pli 0, sous MAEST.
Candidat V2 : MAEST bloc 12 + sonde logistique
([fiche](../reports/v2/candidate.json)). Détail : [comparateurs](../reports/v2/cv_results.csv),
[CNN](../reports/v2/cnn_v2_seeds.csv), [ViT natif](../reports/v2/cv_results.csv),
[ViT affiné](../reports/latest_vit_finetune_seed42_training.json), [candidat](../reports/v2/candidate.json).

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
6. Grille de C élargie (10⁻⁵ à 10).

**Sortie :** tous les candidats comparés en validation croisée ; un seul candidat V2
choisi selon la règle déclarée en phase 2.

## Phase 4 — Robustesse et calibration (≈ 3–4 jours)

**État : terminée le 11 novembre 2026** ([phase 4](../reports/v2/phase4.json),
[résultats](../reports/v2/phase4.json)). Augmentation non adoptée par la règle (0,681) ; elle
résiste pourtant bien mieux au bruit (+0,16 à 10 dB) ; l'auteur a choisi de garder la règle,
la fragilité au bruit fort est une limite déclarée. Température 0,824, seuil 0,337.


**Protocole fixé le 10 novembre 2026, avant tout résultat** (choix de l'auteur : tout le
développement, augmentation de MAEST avec règle d'adoption, rejet à 5 %). Commit du protocole :
`0cc8b5f`, le 10 novembre à 19 h 37, une minute avant le premier calcul de la phase. `reports/v2/phase4.json`,
figé par le verrou 3, le cite sous son identifiant antérieur à la réécriture de l'historique (`2cff310`).
- Conditions ([config](../configs/degradations_v2.json)) : bruit blanc à 30/20/10 dB de SNR sur le
  morceau entier à 16 kHz ; gain numérique ±6 dB sans écrêtage ; MP3 mono 64 kb/s constant
  (libsndfile/LAME, débit mesuré à chaque appel) au taux d'origine, après normalisation de crête.
- Robustesse : les têtes de chaque pli (C déjà choisi) prédisent les 7 857 morceaux dégradés ;
  écart à l'audio propre par bootstrap apparié par artiste (2 000 tirages).
- Augmentation : un tirage par morceau (type uniforme, paramètre uniforme dans les mêmes plages,
  autre graine), ajouté aux ajustements ; choix de C et évaluation sur l'audio propre. **Adoptée
  seulement si la macro-F1 propre en CV dépasse 0,6855** (0,6835 + 0,002, règle du protocole V2) ;
  sinon la tête sans augmentation reste le candidat et le CNN n'est pas réentraîné.
- Calibration par température et seuil de rejet ([config](../configs/calibration_v2.json)) : estimés
  par ajustement croisé sur les plis, puis fixés sur toutes les prédictions hors pli ; 5 % des
  morceaux du développement rejetés. Ni test V2 ni hors domaine.
- Coût estimé : 7 extractions MAEST d'environ 3,8 h (7 processus).

1. Augmentation par bruit, gain et recompression MP3 à l'entraînement des têtes et du CNN.
2. Robustesse mesurée sur tout le développement, pas sur 64 morceaux :
   SNR 30/20/10 dB, ±6 dB de gain, MP3 64 kb/s.
3. Calibration par température sur les plis de validation.
4. Seuil de rejet « genre inconnu » réglé sans toucher l'ensemble hors domaine du test.

## Phase 5 — Évaluation finale V2, une seule fois (≈ 1 jour)

**Verrou 3 posé le 11 novembre 2026** ([verrou 3](../reports/v2/lock3.json),
[verrou](../reports/v2/lock3.json)). **Terminée le 11 novembre 2026** ([évaluation](../reports/v2/final_evaluation/report.json),
[résultats](../reports/v2/final_evaluation/report.json)) : H1 non répliquée, H2 et H3 confirmées ;
recomptage indépendant concordant. Ne plus relancer `--final`.

Inférence unique sur le test V2 et l'ensemble hors domaine, recomptage indépendant,
rapport des trois hypothèses, y compris les résultats défavorables.

## Phase 6 — Interprétation (≈ 2–3 jours)

**Terminée le 11 novembre 2026** ([rapport](INTERPRETATION_V2.md),
[interprétation](INTERPRETATION_V2.md)) : analyse, réplication et écoute de 18 extraits par
l'utilisateur.

1. Analyse par artiste et par genre, et explication de l'écart V1 validation/test.
2. Écoute annotée, par l'auteur, des erreurs avec la fiche existante.
3. Comparaison V1/V2 présentée comme une réplication, pas comme un gain sur le test V1.

## Phase 7 — Livraison (≈ 2–3 jours)

**Points 1, 2 (sur ce poste) et 4 faits le 11 novembre 2026** ([contrôle final](CONTROLE_FINAL.md),
[contrôle final](CONTROLE_FINAL.md)). Restent à l'auteur : licence du code, publication, vidéo,
réinstallation sur un autre poste.

1. Documentation réduite à : README, fiche des données, fiche du modèle, rapport V2,
   architecture, reproduction, index. L'historique V1 va dans `docs/archive/`,
   sans suppression.
2. Réinstallation testée depuis zéro sur un autre compte ou un autre poste.
3. Publication du dépôt (code et rapports, jamais l'audio FMA ;
   licences des poids vérifiées) et vidéo de démonstration.
4. Tableau final de contrôle : chaque point de la critique → corrigé, mesuré ou limite assumée.

## Ce qui dépend de l'auteur

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
