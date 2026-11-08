# Plan révisé — classification musicale informée par l’analyse audio

**État final au 7 novembre 2026 :** campagne V1, notebooks, démo locale et session autonome vérifiés ; bilan général rédigé. CNN utilisateur : 0,5707 de macro-F1 validation, époque 26 sur 30. Voir [la conclusion](CONCLUSION_PROJET.md). Les paragraphes datés ci-dessous conservent les étapes historiques ; réinstallation réseau, publication publique, vidéo et écoute annotée non réalisées.

Version 2 — 19 octobre 2026. Ce document remplace le premier plan sur les choix qu’il révise. Base scientifique : [revue bibliographique](REVUE_BIBLIOGRAPHIQUE.md).

Avancement au 29 octobre 2026 : références descripteurs/ViT/CNN, fusion, ablations et combinaison bloc 6 + HPSS réalisées. Voir [l’index des résumés](resumes/INDEX.md) et la [synthèse vérifiée](../reports/SYNTHESE_VALIDATION.md). Le candidat principal est **bloc 6 + HPSS : macro-F1 0,5975, accuracy 0,6128** sur validation, devant bloc 12 HPSS (0,5844) et bloc 6 original (0,5780). CNN plafond vingt : 0,5407 ± 0,0105 sur trois graines. La fusion initiale n’apporte pas de gain sur validation.

La [revue des régressions](../reports/REGRESSIONS_BLOC6_HPSS.md) confirme 56 corrections et 44 régressions face à bloc 12 HPSS. Pop reste faible (13/94 corrects), Instrumental recule. Le [diagnostic ciblé](DIAGNOSTIC_AUDIO.md) et l’[analyse temporelle](ANALYSE_TEMPORELLE.md) portent sur les trois ViT antérieurs ; ils ne sont pas une nouvelle analyse temporelle de la combinaison. Les [19 fiches d’écoute](ERREURS_VALIDATION.md) restent sans observations subjectives.

Le jalon 9, l’évaluation finale V1, est terminé et vérifié. T1–T6 sont vérifiées : inférence des neuf exports, sensibilité au bruit examinée, bootstrap par artiste contrôlé et [manifeste final verrouillé](../reports/VERROUILLAGE_FINAL.md). Le candidat HPSS est conservé avec sa fragilité au bruit explicitement acceptée ; aucun gagnant global en robustesse n’a été établi. Aucun nouvel entraînement ni extension spécialisée n’est lancé.

T7 terminée : neuf modèles sur 778 morceaux test, 7 002 prédictions contrôlées et 2 000 tirages vérifiés. Candidat : macro-F1 0,5035, accuracy 51,16 % ; gain principal apparié +0,0394, IC [+0,0054 ; +0,0715]. Durée réelle d’inférence 51 min 50 s. [Bilan complet](../reports/EVALUATION_FINALE.md), [fiche modèle](FICHE_MODELE.md). Livraison vérifiée au 6 novembre : trois notebooks exécutés, démo locale, environnements isolés et trame orale portfolio. Restent la session complète utilisateur et son bilan ; réinstallation réseau sur une autre machine, vidéo et écoute annotée non réalisées. [Guide de livraison](GUIDE_LIVRAISON.md). La session utilisateur doit précéder la conclusion générale. [Registre des décisions](DECISIONS_EXPERIMENTALES.md).

FMA-small et ses métadonnées sont désormais téléchargés, vérifiés et extraits. Consulter [l’index de reprise](resumes/INDEX.md) et le [résumé des données](resumes/04_DONNEES.md) pour les preuves et les prochaines actions.

## Objectif

Construire et évaluer un système de classification de huit genres FMA-small utilisant un Vision Transformer. Contribution principale : mesurer si des descripteurs explicites de timbre, de rythme et d’harmonie complètent les représentations apprises par le ViT, et à quel coût.

Livrables : code reproductible, revue sourcée, notebooks d’analyse musicale, résultats comparatifs, fiche du modèle, démo locale et vidéo portfolio. Pas de promesse d’un score fixé à l’avance.

## Contraintes et décisions

### Plan évolutif — décision du 21 octobre 2026

Le plan est une base de travail révisable selon les résultats. Le budget initial
de deux extractions d’ablation est consommé ; il n’interdit pas une extension
justifiée. Avant une nouvelle expérience, documenter l’observation qui la motive,
l’hypothèse, la comparaison contrôlée, le coût estimé et le critère de décision.
Conserver aussi les résultats négatifs et l’historique des choix.
Le [registre des décisions et pistes](DECISIONS_EXPERIMENTALES.md) relie les
observations aux expériences envisageables, sans les présenter comme réalisées.

L’extension CNN à vingt époques, motivée par les époques initiales 9/10/8, est
terminée : gain moyen +0,0054 porté uniquement par la graine 42. Conserver les
deux budgets et prioriser l’analyse des erreurs ; aucune nouvelle prolongation
n’est lancée sans hypothèse supplémentaire.
La combinaison bloc 6 + HPSS est désormais évaluée : gain de macro-F1 +0,0131 face au bloc 12 HPSS et +0,0195 face au bloc 6 original. Les régressions par genre et les erreurs communes sont documentées. Elle devient le candidat principal de la préparation finale, sans prétendre être meilleure pour chaque genre.
La recommandation actuelle est de préparer robustesse et protocole final, plutôt que multiplier les variantes sur la même validation. L’affinage et AST restent des extensions facultatives, non lancées ; une nouvelle observation peut rouvrir cette décision avant le verrouillage.
Les nouvelles explorations utilisent train/validation. Le test reste réservé à
une version finalisée ; une amélioration ultérieure ne réutilisera pas ce test
comme s’il constituait encore une évaluation indépendante de la sélection.

### Décisions initiales

- Environ 8–10 heures de travail par semaine ; viser 12 semaines. La réduction à 8 semaines retire les extensions, pas les contrôles de données.
- Snapdragon X / Windows ARM64, 16 Go de RAM déclarés. Aucun accès CUDA présumé, aucun cloud payant prévu. Tester d’abord l’environnement et chronométrer les traitements ; les durées d’entraînement ne sont pas encore connues.
- FMA-small, genres d’origine, un genre principal. Les paroles, le gospel, le multigenre et la génération musicale sont hors de la première version.
- Python, PyTorch/timm, librosa, scikit-learn et Gradio. Figer les versions après vérification de compatibilité ; figer aussi les révisions des poids préentraînés.
- Conserver les commandes de préparation, extraction, entraînement, évaluation et prédiction séparées. Les notebooks appellent les mêmes modules que la démo.

## Jalons

| Semaines | Travail | Critère de sortie |
|---|---|---|
| 1–2 | Approfondir six lectures prioritaires ; tester outils et calcul ; auditer FMA | Questions expérimentales écrites, environnement testé, partitions figées |
| 3 | Analyser timbre/rythme/harmonie avec écoute et visualisations | Notebook expliquant ce que chaque représentation conserve ou perd |
| 4 | Extraire les caractéristiques et entraîner les références | Référence majoritaire et référence musicale évaluées sur validation |
| 5–6 | Construire la référence CNN et le ViT gelé | Résultats comparables et coût mesuré |
| 7–8 | Évaluer la fusion et les ablations retenues | Tableau répondant aux questions principales, sans recours au test |
| 9 | Verrouiller les configurations, répéter les entraînements concernés, évaluer le test | Rapport avec erreurs, incertitudes et coûts |
| 10 | Réaliser la démo et les tests de bout en bout | Dépôt audio → prédiction traçable |
| 11–12 | Reproduction propre, documentation, vidéo ; extension seulement si marge | Version portfolio installable et conclusions étayées |

## Pipeline et expériences

### Données et audio

Utiliser les partitions officielles après audit des artistes et doublons. En cas de recouvrement avéré, créer une partition corrigée par groupes et conserver le manifeste exact ; distinguer les résultats des benchmarks officiels.

Pour le pipeline ViT : mono, 16 kHz, fenêtres fixes de 10 secondes, log-Mel à 128 bandes, FFT 1 024 et pas 256 comme configuration de référence. Examiner 16 contre 22,05 kHz dans le notebook sur les mêmes extraits ; cette inspection ne constitue pas une preuve de gain de classification. Les checkpoints spécialisés utiliseront exclusivement leur prétraitement documenté.

Fenêtres FMA : [0,10], [10,20], [20,30] secondes, avec zéro-padding si nécessaire. Cache sur disque chargé par lots ; aucune obligation de conserver l’ensemble des spectrogrammes en RAM. Manifeste avec identifiant du morceau, artiste, partition, classe, état du fichier, fenêtres et empreinte de configuration.

### Références et hypothèse principale

1. Prédicteur majoritaire.
2. Descripteurs : moyenne/écart-type de 20 MFCC, centroïde, largeur et contraste spectraux, RMS, chroma à 12 classes, et profil d’autocorrélation d’attaques sur une grille fixe de 40 à 240 BPM par pas de 5. Standardisation apprise sur train, puis régression logistique avec C dans {0,1 ; 1 ; 10}, choisi sur validation.
3. Petit CNN sur le même log-Mel : trois blocs convolution/ReLU/pooling, 16/32/64 canaux, pooling global et tête à huit classes. Il contextualise le transfert du ViT ; entraîné de zéro, il n’isole pas à lui seul l’effet de l’architecture.
4. ViT-Tiny `vit_tiny_patch16_224.augreg_in21k_ft_in1k`, gelé, spectrogramme redimensionné en 224 × 224 et répété sur trois canaux. Extraire la moyenne des tokens de patches du dernier bloc, standardiser sur train et entraîner la même famille de classificateur linéaire.
5. Fusion : concaténer le vecteur ViT aux descripteurs explicites, standardiser chaque dimension sur train et utiliser le même classificateur et la même grille. Aucun gain n’est présumé.

Pour les systèmes par fenêtre, entraîner avec l’étiquette du morceau puis moyenner les probabilités des trois fenêtres à l’évaluation. Figer l’ordre des classes dans l’artefact exporté.

### Ablations, sans grille combinatoire exhaustive

Sur validation, comparer séparément à la référence ViT : fenêtre centrale seule contre trois fenêtres ; bloc 6 contre bloc 12 ; canaux original/harmonique/percussif contre répétition du log-Mel. Pour la fusion, retirer successivement les familles rythme et harmonie. Conserver les mêmes morceaux et les mêmes budgets de recherche ; chaque expérience change un facteur identifié.

La variante HPSS est expérimentale : recalculer les représentations avec ce nouvel agencement de canaux, et documenter le décalage avec les images RGB du préentraînement. Ne pas attribuer automatiquement un gain à une meilleure compréhension de l’harmonie.

Chronométrer 64 fenêtres avant chaque extraction coûteuse. Priorité au noyau des cinq systèmes ; réserver au maximum deux extractions supplémentaires aux ablations. Reporter les autres plutôt que réduire silencieusement les données d’évaluation. Le petit CNN et l’affinage sont plafonnés à 12 heures par expérience, avec checkpoints et reprise ; signaler explicitement un arrêt imposé par le budget.

### Extensions par priorité

1. Affiner les deux derniers blocs du ViT, si le benchmark prévoit dix époques en moins de douze heures ; tester alors un SpecAugment modéré contre son absence.
2. Comparer AST préentraîné, gelé, avec sa représentation audio native. Mesurer les différences de taille, préentraînement, résolution et coût ; ne pas présenter cette comparaison comme une ablation à facteur unique.
3. Étudier un extracteur musical MAEST ou MERT. MuQ et MusicFM restent dans la revue ; exclure MusicFM-FMA du classement principal sur FMA-small en raison de l’exposition au corpus.

## Évaluation et démo

Macro-F1 par morceau comme critère principal ; accuracy, F1 par genre, confusion, mémoire et latence comme mesures complémentaires. Trois graines pour les entraînements stochastiques finaux ; bootstrap groupé par artiste pour les intervalles sur test. Une sélection finale sur validation précède l’ouverture du test. Les analyses de robustesse sont définies avant ce verrouillage.

Tests : intégrité des partitions, normalisations limitées à train, gestion stéréo/fréquences variables/courts extraits/silence/fichiers corrompus, sauvegarde–rechargement, ordre des classes et cohérence entre commande et interface.

Démo Gradio locale (à réaliser ; suivre la [fiche modèle](FICHE_MODELE.md)) : WAV/MP3, maximum 50 Mo et 5 minutes ; trois fenêtres début/milieu/fin pour les longs morceaux, positions affichées. Montrer audio, spectrogramme, top-3 des huit classes, durée de calcul et version. Une fonction commune fournit aussi un JSON. Aucun score affiché comme certitude ; aucune prétention à reconnaître les genres hors vocabulaire. Supprimer les fichiers temporaires après traitement.

La livraison finale comporte les commandes de reproduction, configurations, manifeste, résultats, limites, références et une vidéo de 2–3 minutes. L’hébergement public reste distinct du fonctionnement local et dépendra du budget retenu.

## Ajout demandé — session autonome avant le bilan général

Après les travaux en cours, préparer un environnement virtuel distinct et un
notebook d’entraînement que l’utilisateur exécutera lui-même. Prévoir un plafond
de vingt ou trente époques, une patience explicite, un budget chronométré et des
sorties durables. Les trois CNN ont déjà eu un plafond vingt avec arrêt anticipé ;
la nouvelle recette devra préciser ce qu’elle change. Le ViT gelé n’est pas un
entraînement par époques ; son affinage serait une expérience différente.

À la fin annoncée par l’utilisateur, vérifier les artefacts, extraire les résultats
et mettre à jour le notebook avant la conclusion générale. Si le test V1 a déjà
été consulté, conserver cette extension comme développement complémentaire, sans
revendiquer une nouvelle évaluation indépendante sur ce même test.
Voir [le déroulement et les sorties prévues](SESSION_ENTRAINEMENT_UTILISATEUR.md).
