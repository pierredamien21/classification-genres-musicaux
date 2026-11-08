# Configurations

`cnn_extension.json` fige la prolongation contrôlée des graines 42/43/44 : plafond
vingt époques, checkpoints parents et sources par empreinte, état d’optimiseur et
patience conservés. Les autres paramètres restent ceux de `cnn.json`. La reprise
se fait dans de nouveaux dossiers ; le protocole initial n’est pas remplacé.
Voir `docs/EXTENSION_CNN.md`.

`cnn_repetitions.json` fixe les graines 42/43/44, la référence historique, les
empreintes du protocole et la statistique de dispersion avant les répétitions.
`--seed` ne change que la graine de `cnn.json` ; ses autres paramètres restent
identiques. Aucun choix de meilleure graine ni ensemble implicite n’est prévu.
Voir `docs/REPETITIONS_CNN.md`.

`reference.toml` décrit les décisions du plan, sans déclencher d’entraînement. C’est un brouillon expérimental ; il n’est pas encore consommé par un entraînement.

`diagnostic.toml` pilote le benchmark de faisabilité CPU sur 64 extraits d’entraînement. Ses paramètres audio et dB ne sont pas présentés comme optimaux. `vit_checkpoint.json` conserve la révision et les empreintes attendues des poids téléchargés.

`audit.json` fixe les limites de décodage et seuils de signalement technique avant
l’audit intégral. Les seuils de durée, silence et niveau sont des heuristiques de revue,
pas des motifs automatiques d’exclusion. Ils ne sont pas optimisés sur les genres ou
les résultats d’un modèle.

`review_policy.json` fixe les seuils d’empreintes perceptuelles et documente le profil
de sélection `conservative_baseline_v1`. Ce profil est implémenté dans
`src/music_genre/data/review.py` ; la section textuelle `decisions` n’est pas un
interpréteur de règles. Toute évolution des décisions doit versionner le profil et
son code. La configuration conserve également les références et empreintes de
l’outil officiel Chromaprint.

`descriptors.json` fixe le prétraitement et les 125 descripteurs de la référence musicale.
`baseline_descriptors.json` fixe la grille C, le solveur et la sélection sur validation.
Le cache d’extraction dépend de la configuration des descripteurs, pas de la grille du
classifieur : entraîner un autre C ne nécessite pas de redécoder les morceaux.
Les champs textuels de description documentent le contrat v1 ; changer sa définition
nécessite une modification et une nouvelle version du code d’extraction.

`labels.json` fixe l’ordre interne des huit classes. Les identifiants numériques FMA devront être rapprochés de ces noms à partir des métadonnées officielles, sans supposer qu’ils valent 0 à 7.

`vit_frozen.json` fixe les trois fenêtres, le log-Mel, la plage dB et le checkpoint
vérifié pour les 192 caractéristiques ViT. La plage dB reprend le diagnostic sans
optimisation sur validation. `baseline_vit.json` reprend exactement la grille et
le protocole des descripteurs ; seul le nom de l’expérience change. Les champs
descriptifs de resize, canaux, pooling et normalisation documentent l’implémentation
v1 : les modifier nécessite aussi de modifier le code correspondant.

`baseline_fusion.json`, `baseline_fusion_no_rhythm.json` et
`baseline_fusion_no_harmony.json` gardent la même grille et le même protocole.
L’assemblage conserve les valeurs brutes ; la standardisation est apprise seulement
pendant l’entraînement. L’ordre ViT puis descripteurs et les colonnes retenues sont
consignés dans chaque rapport de cache, avec les configurations et empreintes des
deux parents. Voir `docs/FUSION.md` pour les ablations et les commandes.

Chaque expérience conservera une copie complète de sa configuration, son empreinte, le manifeste utilisé et la révision exacte des poids. Pas d’héritage implicite de configurations dans la première version. Les chemins sont relatifs à la racine du projet, jamais au répertoire d’un notebook.

`cnn.json` fixe la référence CNN de développement : log-Mel natif 128 × 622,
normalisation dB fixe, AdamW, batch 16, quatre fils, graine 42, dix époques maximum
et patience trois. L’architecture 16/32/64 avec premier stride 2 est codée dans
`SmallAudioCNN` ; ses champs descriptifs ne sont pas un constructeur générique.
Le benchmark précède l’extraction et l’entraînement. Les paramètres audio seuls
entrent dans la configuration du cache ; les réglages d’apprentissage restent
dans l’expérience. Une reprise refuse toute modification de son identité
(configuration, code, cache, versions et rapport de benchmark). Voir `docs/CNN.md`.

Les modèles audio spécialisés auront leur propre prétraitement documenté ; `reference.toml` concerne le parcours ViT image.

`baseline_vit_central.json` reprend la grille de la référence ViT. Le cache central
conserve le prétraitement et le checkpoint parent avec `offsets_seconds: [10.0]`.
Le sélecteur exige une centrale par morceau et ne modifie aucune valeur des
embeddings. Le scaler est réappris uniquement sur le train central. Voir
`docs/ABLATION_FENETRES.md` pour la distinction entre ablation principale et
diagnostic complémentaire à modèle fixe.

`vit_block6.json` reprend les paramètres de la référence et fixe le sixième bloc,
avec la même LayerNorm finale gelée du checkpoint et la moyenne des patches sans
préfixes. Les log-Mel CNN servent uniquement de cache d’entrée vérifié, sans changer
le signal. `baseline_vit_block6.json` conserve la grille logistique d’origine.
Voir `docs/ABLATION_BLOCS.md` pour la convention de normalisation et le benchmark.

`vit_hpss.json` conserve le bloc 12 et remplace les canaux répétés par les log-Mel
original/harmonique/percussif. Les paramètres HPSS sont explicites : domaine STFT
complexe, filtres médians 31/31, puissance des masques 2, marges 1/1. Le gain et la
référence dB restent communs aux canaux. `baseline_vit_hpss.json` conserve la grille
logistique initiale. Voir `docs/ABLATION_HPSS.md` pour les contrôles d’équivalence.


`error_review.json` définit la revue ciblée sur validation : sept catégories, quota de trois morceaux, sélection déterministe et un artiste au plus par échantillon global. Les quotas incomplets sont rapportés. Voir `docs/ERREURS_VALIDATION.md`.

`audio_diagnostic.json` fixe les trois ViT, les tolérances de comparaison et
la sélection des figures avant le recalcul des 19 extraits existants. Validation
uniquement, sans réentraînement. Voir `docs/DIAGNOSTIC_AUDIO.md`.

`temporal_analysis.json` fixe les 780 morceaux de validation, les trois ViT,
les positions 0/10/20 s et la tolérance de reproduction des probabilités avant
l’analyse descriptive. Aucun modèle ni agrégation n’est ajusté.
Voir `docs/ANALYSE_TEMPORELLE.md`.

`vit_block6_hpss.json` combine les canaux HPSS existants et le sixième bloc
gelé avec la même LayerNorm finale. `baseline_vit_block6_hpss.json` conserve
la grille logistique des comparateurs ; aucun classificateur candidat n’est
encore entraîné. `block6_hpss_experiment.json` fixe les comparaisons et le
benchmark train avant l’extraction. Voir `docs/COMBINAISON_BLOC6_HPSS.md`.
