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

## Amendement 1 — remplaçants Pop (8 novembre 2026, avant le verrou 2)

**Constat technique.** L'audit a écarté 45 des 80 morceaux Pop tirés, remplaçants
compris : ce sont des extraits de 0 à 15 s dans `fma_large`, alors que la V1 exige 29 à
31 s ([audit](../reports/v2/audit.csv)). Pour 79 de ces 80 morceaux, la durée des
métadonnées (au moins 30 s) prédisait la réussite ; la seule exception est un MP3
indécodable. Sans amendement, Pop n'aurait que 35 morceaux.

**Règle ajoutée.** Les 49 artistes Pop encore disponibles (medium/large, absents de
FMA-small et de toutes les cohortes V2) ayant au moins un morceau d'au moins 30 s
selon les métadonnées sont classés au hasard avec une nouvelle graine (20261109).
Pour chaque artiste, un morceau d'au moins 30 s est tiré, avec la même fonction que la
sélection d'origine ([script](../scripts/amend_v2_pop.py),
[liste](../reports/v2/cohorts_amendment.csv), [trace](../reports/v2/amendment.json)).
Ils deviennent des remplaçants Pop de rangs 80 à 128, utilisés après les remplaçants
d'origine et selon la même règle d'audit.

**Garanties et limite.** La décision repose sur un critère technique, avant tout
modèle V2 et sans aucune prédiction ni écoute. Limite : les morceaux Pop ajoutés
proviennent d'un vivier filtré par la durée, contrairement aux autres genres.
International reste à 53 morceaux : il n'y a plus d'artiste International inutilisé.

### Note d'audit — messages du décodeur (avant le verrou 2)

Après l'audit, 36 fichiers écrivent des messages de libmpg123 au décodage. 20 d'entre
eux sont dans les cohortes finales, dont 15 morceaux International du test, aux numéros
consécutifs. Ce sont surtout des trames MP3 rejetées : `part2_3_length … too large`,
`dequantization failed`. Ces fichiers ont la bonne durée, et fpcalc (ffmpeg) les décode
sans diagnostic ([liste](../reports/v2/decoder_messages.csv),
[script](../scripts/v2_decoder_messages.py)).

**Décision.** La règle d'audit préenregistrée ne prévoit pas d'exclusion sur ce critère :
**on la garde.** Exclure ces fichiers après les avoir vus ferait tomber International
à 38 morceaux, sans remplaçant disponible.

**Analyse de sensibilité déclarée avant tout résultat.** Elle est descriptive et sans
valeur de décision. H1 et le macro-F1 de chaque système seront aussi rapportés sans
ces 20 morceaux.
