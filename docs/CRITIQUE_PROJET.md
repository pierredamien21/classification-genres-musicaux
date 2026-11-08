# Critique complète du projet

8 novembre 2026. Revue externe de bout en bout : code, configurations, données,
résultats stockés et littérature. Aucun code ni résultat existant n'a été modifié.

## Verdict en bref

Le projet est **très rigoureux sur la procédure** (partitions par artiste, test
verrouillé, provenance, bootstrap par artiste, résultats négatifs conservés) mais
**faible sur la portée scientifique** : le « Vision Transformer » n'est qu'un
extracteur ImageNet gelé suivi d'une régression logistique, comparé à un CNN
volontairement minuscule, sans aucun modèle audio préentraîné. Le résultat test
(macro-F1 0,5035, accuracy 51,2 %) est honnête mais en dessous de ce que la
littérature obtient sur FMA-small (souvent 55–65 %, protocoles hétérogènes).
Enfin, l'évaluation repose sur très peu d'artistes : l'écart validation→test,
présenté comme inexpliqué, s'explique en grande partie par quelques artistes.

## Ce que fait réellement le système

Audio mono 16 kHz, trois fenêtres de 10 s (0/10/20 s). Log-Mel 128 × 622,
séparation HPSS sur la STFT complexe, trois canaux original/harmonique/percussif,
bornage −80/+20 dB, redimensionnement bicubique en 224 × 224, ViT-Tiny ImageNet
gelé tronqué au bloc 6, moyenne des tokens de patches (192 dimensions),
standardisation et régression logistique (C ∈ {0,1 ; 1 ; 10}), moyenne des
probabilités des fenêtres. Voir [hpss.py](../src/music_genre/features/hpss.py),
[vit_channels.py](../src/music_genre/models/vit_channels.py),
[train_descriptor_baseline.py](../scripts/train_descriptor_baseline.py).

## Points forts

- **Hygiène d'évaluation exemplaire pour un portfolio** : disjonction des artistes
  vérifiée par le code ([final_protocol.py](../src/music_genre/evaluation/final_protocol.py)),
  normalisation apprise sur train uniquement, test ouvert une seule fois après
  verrouillage par empreintes, bootstrap apparié **par artiste**.
- **Honnêteté** : la fusion qui n'aide pas, la chute sur test, la fragilité au
  bruit et la session postérieure au test sont toutes signalées comme telles.
- **Audit des données** réel : doublons PCM exacts, empreintes Chromaprint,
  décisions traçables pour les 8 000 morceaux.
- **Code défensif et testé** : 271 tests réussis en 26 s (relancés le 8 novembre),
  validation systématique des formes, des valeurs finies et de l'ordre des classes.
- **Ablations propres** : fenêtres, profondeur (bloc 6 > bloc 12), canaux HPSS,
  familles de descripteurs, trois graines CNN.

## Critiques scientifiques, par ordre d'importance

### 1. La question « ViT » n'est pas vraiment testée

Le ViT n'est jamais affiné, et aucun modèle préentraîné **sur de l'audio** n'est
comparé. La [revue bibliographique](REVUE_BIBLIOGRAPHIQUE.md) désigne pourtant AST
comme « le comparateur naturel » et une « extension prioritaire » ; PANNs, MAEST,
MERT y sont discutés mais jamais mesurés. Sans ce point de comparaison, on ne peut
pas dire si 0,50 est bon pour une représentation gelée : on sait seulement que ce
ViT image bat un CNN de 24 000 paramètres et des descripteurs classiques.

Repères publiés (protocoles non comparables, à lire comme ordre de grandeur) :
BYOL-A en embeddings ≈ 64,3 % sur FMA-small (découpage non précisé) ; ResMamba
64,5 % en validation croisée par morceau ; une étude de 2025 relève 48 à 80 % selon
les articles, avec des découpages souvent non précisés ou aléatoires, qui peuvent
laisser le même artiste en entraînement et en test.
Le projet, lui, respecte le découpage par artiste : ses chiffres sont plus fiables,
mais le modèle reste nettement sous l'état de l'art.

### 2. L'évaluation repose sur une poignée d'artistes

Recalcul à partir des prédictions stockées :

| Partition | Morceaux | Artistes | Part des 10 plus gros artistes |
|---|---:|---:|---:|
| Train | 6 299 | 1 875 | 9 % |
| Validation | 780 | 219 | 35 % |
| Test | 778 | 202 | **44 %** |

En test, un seul artiste Folk fournit **91 des 100 morceaux Folk** ; Folk et
Instrumental n'ont que 9 et 11 artistes. Le F1 « Folk » mesure donc surtout cet
artiste. Exemples pour le candidat :

| Partition | Artiste (genre) | Morceaux | Accuracy |
|---|---|---:|---:|
| Validation | 12271 (Hip-Hop) | 55 | 0,89 |
| Validation | 21803 (Instrumental) | 47 | 0,77 |
| Test | 77 (Folk) | 91 | 0,36 |
| Test | 19472 (Instrumental) | 51 | 0,24 |

Conséquence : **l'écart validation→test n'est pas propre au candidat**. Tous les
systèmes chutent : 9,4 points pour le candidat, 9,1 en moyenne pour les huit
autres (de 4,3 pour la fusion à 12,9 pour le CNN graine 44). C'est surtout un
effet de composition des partitions, pas d'abord un surapprentissage de la
sélection. Le classement est d'ailleurs conservé (le candidat reste premier).
Les rapports qui disent « cause non établie » peuvent être précisés.

### 3. Les intervalles de bootstrap du macro-F1 sont décalés

Dans [bootstrap_draws.npz](../reports/final-test-report-v1/bootstrap_draws.npz),
la moyenne des tirages du macro-F1 est **inférieure d'environ 2,3 points** à la
valeur ponctuelle pour chaque modèle, et 74 % des tirages sont en dessous
(candidat : 0,5035 ponctuel, 0,4803 en moyenne). D'où des intervalles
asymétriques comme [0,4133 ; 0,5449]. Cause probable : le macro-F1 est non
linéaire et le rééchantillonnage d'artistes très inégaux déséquilibre les
classes. L'accuracy n'a pas ce biais.

Ce qui tient : les **différences appariées ne sont pas biaisées** (candidat − ViT :
+0,0394 ponctuel, +0,0388 en moyenne, 1,3 % de tirages ≤ 0). La conclusion
principale est donc solide ; seuls les intervalles individuels du macro-F1 sont
mal centrés. Un intervalle BCa ou la mention explicite du biais corrigerait cela.

### 4. Le CNN de comparaison est trop faible pour conclure

[SmallAudioCNN](../src/music_genre/models/cnn.py) : trois convolutions 16/32/64,
environ 24 000 paramètres, sans normalisation de batch, sans dropout, sans
augmentation, patience 3 sur une validation qui oscille de ±2 points d'une époque
à l'autre. La session utilisateur l'a confirmé : avec patience 5 et 30 époques,
la même graine gagne 2,4 points. « ViT gelé > CNN » ne dit donc rien de la
comparaison Transformer/CNN en général.

### 5. Le passage spectrogramme → image est discutable et non testé

128 × 622 est redimensionné en 224 × 224 : le temps est comprimé ×2,8 et la
fréquence étirée ×1,75 ; chaque patch 16 × 16 couvre environ 0,7 s × 9 bandes Mel.
AST garde la résolution native et interpole les positions. Le bornage fixe
−80/+20 dB, sans normalisation par morceau ni augmentation, explique sans doute
une partie de la fragilité mesurée : −0,18 de macro-F1 à SNR 20 dB (panel de 64
morceaux). Aucune variante d'entrée n'a été essayée.

### 6. Le classifieur est limité en capacité et la grille de C est trop étroite

C = 0,1, borne basse de la grille, est retenu pour 5 des 6 systèmes linéaires.
Les écarts entre valeurs de C sont inférieurs à 1 point, donc l'impact est faible.
Mais le macro-F1 train du candidat n'est que de 0,635 : le modèle sous-apprend.
La vraie marge est dans la représentation ou une tête non linéaire, pas dans C.

### 7. Taxonomie et couverture

Les genres les plus mal classés en test (Pop 0,33, Experimental 0,36, Folk 0,37,
Instrumental 0,42) incluent des catégories qui ne sont pas des genres acoustiques (« Experimental »,
« Instrumental », « International »). L'objectif initial (rap, gospel) n'est
couvert que partiellement, ce que le projet reconnaît. Le test compte 778 morceaux
au lieu de 800 (quarantaine technique) : comparaison directe avec le découpage
officiel impossible, ce qui est aussi signalé.

## Critiques d'ingénierie

- **Pas de dépôt Git.** Le dossier a un `.gitignore` et la documentation parle de
  fichiers « exclus de Git », mais `git status` échoue. Les empreintes SHA-256
  prouvent qu'un fichier n'a pas changé ; elles ne remplacent pas l'historique.
  *Corrigé le 8 novembre : dépôt créé, étiquette `v1-final`.*
- **Sur-ingénierie.** 9 500 lignes de scripts pour 2 300 lignes de bibliothèque.
  Quatre scripts `extract_vit*.py` et trois `train_*cnn*.py` sont presque des
  copies. Les variantes passent par des sous-classes qui vérifient des chaînes de
  configuration et tronquent le modèle après coup ;
  [hpss_block6.py](../src/music_genre/features/hpss_block6.py) transmet même une
  fausse valeur de `pooling` à la classe parente. Un extracteur paramétré
  (canaux, bloc) suffirait.
- **Documentation trop abondante et parfois contradictoire.** 35 documents,
  40 résumés, 142 fichiers dans `reports/`. Exemples d'incohérences :
  le [README](../README.md) annonce 261 puis 271 tests dans le même paragraphe ;
  la [fiche du modèle](FICHE_MODELE.md) dit « interface non livrée » alors que la
  démo existe ; l'[index](resumes/INDEX.md) affiche « Mis à jour : 29 octobre »
  pour un contenu du 7 novembre, et des lignes vides cassent son tableau ;
  plusieurs `__init__.py` disent encore « à implémenter ». Un relecteur de
  portfolio aura du mal à trouver l'essentiel.
  *Précision du 8 novembre : la fiche du modèle et les `__init__.py` sont épinglés
  par empreinte dans les preuves V1 (reçu du test, manifeste verrouillé). La fiche
  est un instantané daté, pas une erreur ; ils restent inchangés jusqu'à une V2
  verrouillée. README et index sont corrigés.*
- **Principe non tenu.** Le README promet « une fonction de prédiction partagée
  par l'interface et la CLI », mais la [CLI](../src/music_genre/cli.py) n'a que
  `doctor`.
- **Démo** ([app/main.py](../app/main.py)) : l'audio est décodé deux fois par
  requête. Mineur pour un usage local. *Correction du 8 novembre : une première
  version de cette critique disait que les fichiers envoyés n'étaient jamais
  nettoyés. C'est faux : `delete_cache=(600, 3600)` les purge.*
- **Environnement** : Python x86-64 émulé sur Windows ARM64 et trois
  environnements virtuels distincts ; réinstallation sur une machine vierge non
  testée (reconnu).

## Recommandations, par priorité

1. `git init`, premier commit des sources, configurations et rapports légers.
2. Ajouter un modèle audio préentraîné (AST AudioSet, PANNs ou MAEST) en sonde
   linéaire avec le même protocole. Le test V1 étant consommé, prévoir une
   nouvelle évaluation indépendante (par exemple des artistes de FMA-medium
   absents de FMA-small) ou une validation croisée groupée par artiste.
3. Compléter l'évaluation : analyse par artiste, métriques pondérées par artiste
   ou plafond de morceaux par artiste, intervalles BCa pour le macro-F1.
4. Renforcer le CNN (normalisation de batch, SpecAugment, patience plus longue)
   ou présenter la comparaison comme un simple repère.
5. Tester une entrée à résolution native (interpolation des positions) et une
   augmentation par bruit et gain.
6. Fusionner les scripts dupliqués et réduire la documentation à un README, une
   fiche du modèle, un rapport final et l'index.

## Méthode et limites de cette critique

Vérifié : lecture de toute la bibliothèque `src/`, des configurations principales,
des scripts d'entraînement linéaire et des rapports finaux ; suite de tests
relancée (271 réussis) ; concentration des artistes, accuracy par artiste,
chute validation→test et biais du bootstrap recalculés à partir des prédictions
et tirages stockés (scripts jetables, hors du projet).

Non vérifié : aucune extraction ni aucun entraînement relancé ; les 66 scripts
et les trois notebooks n'ont pas été lus en entier ; pas d'écoute des morceaux.
Les chiffres de la littérature viennent de recherches web et d'extraits
d'articles, avec des protocoles différents de celui du projet.
