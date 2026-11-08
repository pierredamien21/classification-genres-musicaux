# Stratégie de tests

Cent vingt cas sont actuellement implémentés : audio stéréo, rééchantillonnage, fenêtres,
entrées invalides, silence et embeddings ViT déterministes et finis ; audit technique,
identité PCM indépendante du conteneur, doublons entre partitions, attribution des
diagnostics natifs et distinction entre signalements et exclusions ; alignement
d’empreintes, contrôle de signaux peu informatifs et décisions de sélection avec
conflits de labels, répétitions exactes et groupes de candidats ; descripteurs physiques
sur sinus et impulsions, padding, agrégation par morceau et export linéaire numérique ;
contrat audio du ViT, rejet d’un checkpoint modifié et refus des comparaisons dont
les fenêtres, les grilles ou les prédictions ne correspondent pas ; fusion conservant
les valeurs sources et l’ordre des colonnes, retraits de familles, refus des fenêtres
désalignées ou dupliquées, du test, des valeurs non finies et des ablations inconnues ;
CNN avec gradients finis, normalisation fixe, export safetensors identique, permutation
des époques reproductible et reprise exacte de l’optimiseur AdamW sur un exemple contrôlé ;
sélection centrale conservant les morceaux et métadonnées, refus des fenêtres manquantes
ou dupliquées, des labels incohérents, d’un mauvais ordre des lignes et du test ;
bloc 6 égal à une capture du réseau complet suivie de la même LayerNorm, bloc 12
égal à l’extracteur original, et rejet des indices de blocs invalides avant chargement ;
canal original HPSS identique au log-Mel de référence, silence fini, dominance
harmonique d’un sinus et percussive d’impulsions, et entrée à trois canaux identiques
reproduisant le ViT original gelé.
Les répétitions CNN ajoutent le calcul de dispersion entre graines et le refus
des graines dupliquées, des protocoles ou caches différents, des sources modifiées,
des runs incomplets, de l’utilisation du test et des scores non finis.
Les tests nécessitant le ViT réel utilisent uniquement des poids locaux et sont ignorés s’ils sont absents.
L’extension CNN contrôle la continuité exacte des poids, de l’optimiseur et du RNG,
la conservation du parent, le refus des changements de paramètres/données/sources,
la patience, les historiques et l’égalité des meilleurs poids exportés. La boucle
d’apprentissage, sélection et export reste identique à celle du protocole initial.
Sur le poste de développement, les cent vingt cas ont été exécutés avec succès.
Voir `docs/ENVIRONNEMENT.md` pour les commandes et le dossier temporaire local.

Vérifications à compléter au fil de l’implémentation :
- Données : absence de fuite entre partitions, rapprochement des classes et manifeste stable.
- Audio : stéréo, fréquences variables, valeurs non finies, silence, fichier corrompu et fenêtres courtes.
- Représentations : dimensions cohérentes et aucune normalisation ajustée sur validation/test.
- Cache : invalidation après changement de source, modèle, extraction ou normalisation.
- Modèles : correspondance labels/scores et équivalence après sauvegarde/rechargement.
- Évaluation : agrégation par morceau et bootstrap par artiste.
- Intégration : même entrée, mêmes fenêtres et mêmes scores dans CLI et démo.

Employer des signaux synthétiques et de petits exemples autorisés. Les tests rapides n’accèdent ni au réseau ni au dataset complet ; séparer les tests qui exigent des poids ou du calcul.


L’analyse d’erreurs ajoute dix cas : alignement par identifiant malgré l’ordre CSV, refus des lignes test, prédictions manquantes/dupliquées, vérités discordantes, argmax incohérent, probabilités invalides ; décomposition des corrections/régressions ; sélection déterministe par artiste et quotas incomplets explicites.

Le diagnostic audio ajoute cinq cas : RMS calculé hors padding, rejet du test,
d’un index temporel incohérent, d’un cache modifié et de valeurs non finies.

L’analyse temporelle ajoute neuf cas : exemple chiffré de moyenne/vote avec
réordonnancement, refus du train/test, des positions ou identifiants de fenêtres
incorrects, des artistes ou sources incohérents et des probabilités invalides.

La combinaison bloc 6 + HPSS ajoute cinq cas : entrée identique à HPSS,
sortie égale au sixième bloc du réseau complet suivi de sa LayerNorm, et refus
précoce des protocoles incompatibles (bloc 12, booléen, normalisation ou pooling).
Le test avec poids locaux confirme aussi le gel des paramètres.

Le cache par morceau ajoute sept cas : sauvegarde/relecture exacte, refus
d’une empreinte ou identité de configuration/source modifiée, et refus de
dimensions, valeurs non finies ou durées invalides. Le pilote réel a confirmé
la reprise de 64 morceaux sans recalcul.

La consolidation ajoute un cas chiffré de corrections/régressions par genre :
un rappel inchangé peut cacher des morceaux différents, et les faux positifs
doivent être comptés parmi les autres genres. Suite complète : 121 tests réussis
le 28 octobre 2026 ; les rapports réels sont aussi recoupés avec leurs matrices
de confusion et les comparaisons antérieures.

L’inférence du candidat ajoute quatorze cas : fichier stéréo court réellement
traité par les poids locaux, silence, trois bornes de ressources, fichiers
corrompu/vide/non fini/manquant et annulation mono, exports modifiés/manquants,
ordre des classes invalide et chemin hors des artefacts. Total : 135 tests
réussis le 28 octobre 2026. Le script de contrôle séparé reproduit également
les probabilités enregistrées sur 64 vrais morceaux de validation.

La sensibilité audio ajoute neuf cas : identité/gain sans mutation ni écrêtage,
SNR à 20 dB et graine déterministe par morceau, cinq entrées invalides,
padding non bruité et déclencheur de revue par changement de classe.
144 tests réussis le 28 octobre 2026. Le contrôle réel vérifie en plus
64 morceaux dans trois conditions, soit 576 fenêtres.

Le diagnostic des représentations ajoute six cas : variations brutes distinctes
des entrées bornées, distance standardisée sur exemple connu, normes nulles
explicites et refus des formes/échelles/valeurs/bornes invalides. Total : 150 tests
réussis le 28 octobre 2026. Le diagnostic réel reproduit aussi les probabilités
T2 des huit pilotes et conserve 72 embeddings vérifiables.

## Adaptateur bloc 6 original (T2b)

`test_inference_adapters.py` ajoute cinq cas : reproduction d’un morceau de
validation sauvegardé, refus d’un modèle modifié/manquant et d’un chemin de
run extérieur, puis fichier court sans votes dupliqués. Exports locaux requis,
sinon tests explicitement ignorés. Suite actuelle : **155 tests réussis**.
Le vérificateur `scripts/check_block6_robustness_comparison.py` recalcule
indépendamment les métriques du panel et les 192 empreintes PCM ; preuve dans
`reports/BLOCK6_ROBUSTNESS_CHECK.json`.

## Bootstrap par artiste (T3)

`test_bootstrap.py` ajoute 23 cas : multiplicité et groupes inégaux, maintien des
labels absents, alignement et valeurs invalides, différences appariées, moyenne
de métriques CNN, graine et permutation des lignes. Un cas protège la tolérance
10⁻⁶ nécessaire aux probabilités CNN en float32, sans renormalisation.
Suite actuelle : **178 tests réussis**. Le script
`scripts/verify_bootstrap_validation.py` développe les morceaux de chaque tirage
et recalcule avec scikit-learn 44 000 métriques ; écart maximal nul. Preuve :
`reports/BOOTSTRAP_VALIDATION_CHECK.json`.

## Comparateurs linéaires (T4)

`test_linear_comparators.py` ajoute 18 cas : quatre reproductions audio, fusion
mal configurée, artefacts altérés, fichiers courts stéréo et silence, géométrie
exacte en échantillons, puis reproduction sous un appelant BLAS multithread.
Suite actuelle : **196 tests réussis**, avec `--basetemp` dans le workspace.
Le vérificateur `scripts/verify_comparator_inference.py` contrôle les moyennes
des 1 152 fenêtres, les références et les métriques complètes des six systèmes.
Preuve : `reports/COMPARATOR_INFERENCE_RECOUNT.json`.

## Inférence CNN (T5)

`test_cnn_inference.py` ajoute 11 cas : trois checkpoints sélectionnés et
reproduction de leur référence, fichiers stéréo courts, silence, restauration
des threads, modèle altéré, époque incohérente, architecture incompatible,
chemin extérieur et graine non prévue. Suite actuelle : **207 tests réussis**.
Le vérificateur `scripts/verify_cnn_inference.py` contrôle les 576 fenêtres,
les 192 classes du panel et les métriques des trois validations complètes.
Preuve : `reports/CNN_INFERENCE_RECOUNT.json`.
