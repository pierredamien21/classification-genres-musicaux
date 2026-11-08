# Contrats prévus pour les données et résultats

Ces contrats guident l’implémentation progressive. Au 28 octobre 2026, les fonctions
audio, manifestes, caches, exports et comparaisons de validation descripteurs/ViT/CNN/fusion
existent. L’[inférence du candidat](INFERENCE_CANDIDAT.md) depuis un fichier audio
est implémentée et vérifiée sur 64 morceaux ; les autres adaptateurs, l’évaluation
test et la démo restent à produire. Le [protocole final préparé](PROTOCOLE_EVALUATION_FINALE.md)
précise les contrôles préalables et le futur verrouillage ; il n’est pas encore exécuté.

## 1. Classes

`configs/labels.json` contient huit noms dans un ordre stable. L’indice de liste est l’indice de sortie du modèle ; il ne s’agit pas de l’identifiant de genre FMA. Construire et contrôler le rapprochement avec les métadonnées officielles.

Aucune classe « inconnu » n’est entraînée dans la première version. Une faible préférence ou un fichier non musical ne devient pas automatiquement une neuvième classe.

## 2. Manifeste des morceaux

Format initial : CSV UTF-8, une ligne par morceau. Colonnes minimales :

| Champ | Sens |
|---|---|
| track_id | Identifiant FMA conservé comme chaîne, unique dans le manifeste |
| artist_id | Identifiant utilisé pour contrôler les partitions |
| audio_path | Chemin relatif à data/raw, contrôlé pour rester dans ce dossier |
| genre | Nom exact présent dans labels.json |
| split | train, validation ou test |
| duration_seconds | Durée effectivement décodée |
| source_sha256 | Empreinte du fichier source |
| status | valid, needs_review ou excluded |
| exclusion_reason | Motif documenté pour une exclusion |

Un identifiant d’artiste absent doit être traité par une politique explicite avant le split ; ne pas regrouper tous les inconnus sous un faux artiste commun. Une exclusion ne supprime pas la trace du fichier. Le nom de l’artiste, le titre et le chemin ne sont pas des entrées du classificateur.

Le manifeste d’audit est produit par `scripts/audit_fma.py` dans un dossier daté sous
`data/manifests/`, désigné par `latest_audit.json`. Il conserve aussi `label_index`,
`album_id`, `license`, les caractéristiques techniques, `pcm_sha256`, les alertes et
les diagnostics du décodeur. L’empreinte PCM inclut fréquence et nombre de canaux ;
elle compare le float32 décodé natif, sans prétendre détecter les quasi-doublons perceptuels.

`valid` signifie sans alerte selon les contrôles effectués, pas une garantie de qualité.
`needs_review` conserve les fichiers décodables avec alerte, sans exclusion automatique.
Les fichiers non décodables ou entièrement nuls sont `excluded` avec motif. Les limites
techniques et seuils sont fixés dans `configs/audit.json` avant l’exécution et repris
dans le rapport. Les doublons, mêmes partitions ou partitions différentes, restent
en revue. Aucun déplacement entre partitions n’est effectué.

La première sélection est maintenant produite par `scripts/build_baseline_manifest.py`.
`latest_baseline.json` indique un manifeste de lignes retenues et un journal `decisions.csv`
conservant les 8 000 lignes, le statut initial et la décision `keep`/`hold`/`exclude`.
Chaque rapport référence les empreintes de l’audit, du scan, du code et de la politique.
`hold` n’est pas une exclusion définitive. Les répétitions exclues portent le représentant
retenu ; les conflits de genres et candidats non exacts restent en quarantaine.
La construction refuse une duplication PCM retenue ou un artiste/groupe de contenu
partagé entre partitions. Voir `docs/REPRODUCTION_REVUE.md`.

Le CSV est accompagné d’un JSON de provenance : source, date, schéma, checksum, politique de partition, critères d’exclusion, version de code. Le contrôle des doublons audio décodés complète celui des fichiers ; changer les tags MP3 peut modifier l’empreinte sans changer la musique.

## 3. Fenêtres et caractéristiques

Une fenêtre porte `track_id`, `window_id`, `start_seconds`, `valid_seconds` et `padded_seconds`. Les frontières temporelles renvoient au signal décodé, avant padding.

Ne jamais créer de fenêtre entièrement hors du signal. Pour un fichier plus court que dix secondes, utiliser une fenêtre complétée ; ne pas répéter artificiellement trois fois le même extrait. Tout traitement alternatif des morceaux FMA incomplets sera enregistré dans l’audit.

Les matrices de caractéristiques seront stockées en tableaux numériques sans objets Python, accompagnées d’un index de lignes et d’un JSON de provenance. La fusion se fait sur `(track_id, window_id)`, avec contrôle des doublons et des lignes absentes ; jamais sur l’ordre supposé de deux fichiers.

Les paramètres d’extraction, l’ordre des descripteurs, le bloc du modèle, la révision des poids, la normalisation et les empreintes des sources participent à la validité du cache. Un cache d’embeddings gelés ne peut pas être réutilisé après affinage du modèle.

## 4. Expériences et exports

Une exécution conserve configuration, provenance, manifeste, environnement et métriques. Les métriques portent la partition, l’unité d’analyse et le mode d’agrégation ; ne pas mélanger scores par fenêtre et scores par morceau.

Un export doit être autonome pour l’inférence : version de format, labels, configuration de segmentation/prétraitement, scaler éventuel, extracteur ou référence exacte à ses poids, classificateur et provenance. Le chargement échoue explicitement si une pièce indispensable manque.

Les checkpoints de reprise incluront les états du modèle et de l’optimiseur, l’époque, la meilleure métrique et les états aléatoires nécessaires. Une reprise peut rester non strictement déterministe selon les opérations utilisées ; documenter la tolérance.

## 5. Prédiction

La fonction commune à la CLI et à Gradio recevra un chemin audio et un export. Son résultat JSON contiendra : version du modèle, classes ordonnées, scores, top-3, fenêtres réellement analysées et temps de traitement.

La prédiction d’un morceau est la moyenne des probabilités des fenêtres valides. Les fichiers courts ne créent pas de votes supplémentaires par duplication. Un fichier vide, non décodable ou non exploitable produit une erreur explicite, pas un score inventé.

La démo affiche des scores de classification, sans les présenter comme probabilités de vérité calibrées. Les plafonds de taille et durée s’appliquent avant le calcul coûteux, et le décodage devra limiter aussi le volume de données en mémoire.

