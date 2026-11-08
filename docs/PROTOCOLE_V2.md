# Protocole V2 préenregistré

Enregistré le 8 novembre 2026, **avant tout téléchargement d'audio V2**. Version
machine : [configs/protocol_v2.json](../configs/protocol_v2.json). Le commit Git qui
introduit ces fichiers sert de date de dépôt.

## Pourquoi un nouveau test

Le test V1 est consommé, et il était concentré : 10 artistes = 44 % des morceaux,
9 artistes Folk. Le test V2 tire **un morceau par artiste**, avec des artistes absents
de tout FMA-small. Les morceaux viennent de FMA medium/large, ce qui reproduit la
même collection mais sans l'effet artiste.

## Cohortes ([sélection](../reports/v2/selection.json))

| Rôle | Morceaux | Règle |
|---|---:|---|
| Test | 512 (64 × 8 genres) | Ordre d'artistes tiré au sort (graine 20261108), un morceau tiré par artiste |
| Remplaçants test | 115 | Utilisés dans l'ordre, seulement après un échec d'audit technique (International : 3) |
| Hors domaine | 340 | Blues 27, Classical 64, Country 36, Easy Listening 4, Jazz 64, Old-Time 64, Soul-RnB 17, Spoken 64 |
| Remplaçants hors domaine | 64 | Classical, Jazz, Old-Time, Spoken |
| Sonde Gospel | 12 | Morceaux étiquetés Gospel (genre 567) |
| Développement | 7 857 | FMA-small retenu (train, validation et test V1), 5 plis groupés par artiste |

Les artistes sont disjoints entre tous les rôles et de FMA-small (vérifié par le
script). Différence assumée : les étiquettes medium/large peuvent être plus bruitées
que celles de FMA-small. On mesure cet écart, on ne le corrige pas.

## Audit et remplacement, sans regarder les prédictions

Chaque morceau téléchargé passe le CRC-32 du ZIP, l'audit audio de la V1, puis une
recherche de doublons exacts et de quasi-doublons (Chromaprint) contre le
développement et les autres cohortes. Un échec fait appel au remplaçant suivant du
même genre. Ni écoute ni ré-étiquetage.

## Hypothèses

- **H1, primaire (α = 0,05)** : réplication de la V1. Le candidat V1 `vit_block6_hpss`
  bat-il le ViT original sur le test V2, avec les exports V1 inchangés ?
- **H2, secondaire** : le meilleur modèle audio préentraîné gelé contre le candidat V1.
- **H3, secondaire** : le système V2 choisi contre le CNN renforcé (moyenne de 3 graines).
- H2 et H3 : correction de Holm entre elles. Décision : l'intervalle BCa apparié de la
  différence de macro-F1 exclut 0.

**Puissance connue** ([calcul](../reports/v2/power.json)) : la plus petite différence
détectable est de 5 à 7 points de macro-F1. Pour l'effet V1 (+3,9 points), H1 n'a
qu'environ 56 % de puissance. La taille est plafonnée par les 67 artistes International
disponibles ; un échec de H1 ne prouvera donc pas l'absence d'effet.

## Choix des systèmes V2 et mesures

Le système V2 est choisi uniquement en validation croisée sur le développement :
meilleure macro-F1 hors pli moyenne ; en dessous de 0,002 d'écart, le moins coûteux
l'emporte. Calibration, seuil de rejet et robustesse sont réglés ou mesurés sur les
prédictions hors pli du développement, jamais sur le test ou le hors domaine.

## Trois verrous

1. Ce protocole et les listes de cohortes, avant le téléchargement.
2. Les empreintes audio auditées et les cohortes finales, avant tout modèle V2 sur elles.
3. Systèmes, poids, code et environnement, avant l'unique inférence sur le test V2.
