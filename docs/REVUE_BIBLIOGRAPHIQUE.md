# Revue bibliographique — représentations audio et Transformers pour les genres musicaux

Date de recherche initiale : 19 octobre 2026. Complément HPSS : 21 octobre 2026.
Revue ciblée destinée à orienter le projet.

## 1. Objectif et méthode

Question du projet : **quelles représentations du timbre, du rythme et de l’harmonie permettent à un Vision Transformer de distinguer les genres musicaux, avec un coût de calcul accessible ?**

Cette première revue sélectionne des travaux fondateurs et des approches plus récentes, jusqu’à MuQ (2025). Elle ne prétend pas être une revue systématique exhaustive ni établir le meilleur modèle disponible en 2026. Les sources consultées sont des articles, leurs dépôts officiels et les ressources pédagogiques de leurs auteurs. Les articles de génération musicale et de reconnaissance de parole ne sont retenus que si leur méthode apporte quelque chose à la représentation audio.

Requêtes principales : « musical genre classification audio signals », « music genre recognition evaluation artist filter », « audio spectrogram transformer », « music representation learning transformer », « MAEST », « MERT », « MusicFM », « MuQ », « harmonic percussive separation ».

Niveau de consultation : texte de l’article pour Tzanetakis–Cook, Sturm, FMA et MAEST ; passages méthodologiques pour MERT ; notices, résumés, documentation et/ou dépôts pour les autres modèles. Les chiffres de performance ne sont pas repris comme un classement : tâches, données et protocoles diffèrent. Avant de reproduire un modèle, lire intégralement sa méthode et sa configuration officielle.

## 2. Comprendre ce que contient le son

Le spectrogramme est déjà une extraction d’information : il rend explicite l’évolution de l’énergie dans les fréquences. Mais sa résolution, son échelle et sa compression déterminent les indices accessibles au modèle. « Plus de transformations » ne signifie pas automatiquement « plus d’information ».

| Dimension | Représentations à examiner | Information recherchée | Limite à garder en tête |
|---|---|---|---|
| Timbre et texture | Log-Mel, MFCC, centroïde, contraste spectral | Répartition de l’énergie, texture et indices liés aux instruments | Ces indices ne constituent pas une identification fiable d’instruments |
| Rythme | Enveloppe d’attaques, autocorrélation, tempogramme | Régularités temporelles et plusieurs périodicités possibles | Un BPM isolé est ambigu ; deux genres peuvent partager le même tempo |
| Harmonie et hauteurs | CQT, chromagramme | Structure des hauteurs, classes de notes, relations harmoniques | Le chroma agrège les octaves et ne décrit pas à lui seul des accords |
| Composantes du mélange | Séparation harmonique/percussive (HPSS) | Observer séparément continuités tonales et événements transitoires | Ce n’est ni une séparation voix/instruments ni une séparation parfaite |
| Évolution temporelle | Plusieurs fenêtres d’un morceau | Variations entre passages | Moyennes et statistiques perdent une partie de l’ordre temporel |

Les ressources FMP de Meinard Müller expliquent la [STFT](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C2/C2_STFT-Basic.html), le [chroma](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S1_SpecLogFreq-Chromagram.html), le [tempogramme](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S2_TempogramFourier.html) et la [séparation harmonique/percussive](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S1_HPS.html). Elles serviront aux notebooks d’analyse et d’écoute.

Choix à justifier dans notre projet :

- Une longue fenêtre STFT améliore la résolution fréquentielle mais réduit la localisation temporelle. Le pas règle l’espacement entre trames ; il ne résout pas à lui seul ce compromis.
- Passer à 16 kHz retire le contenu au-dessus de 8 kHz. C’est un compromis de calcul et de compatibilité, pas une valeur universellement optimale pour la musique.
- Redimensionner un spectrogramme rectangulaire en carré modifie sa résolution sur les deux axes. Il faut mesurer les conséquences, sans supposer que cela rend nécessairement le modèle mauvais.
- Répéter un canal trois fois assure une compatibilité avec un ViT RGB ; cela ne crée aucune information. Un assemblage original/harmonique/percussif est une hypothèse distincte, qui change aussi la distribution des entrées.
- Une image colorisée ajoute une convention d’affichage. Les modèles recevront des valeurs numériques, sans titres, axes ni légendes.
- Chaque checkpoint audio conservera sa fréquence d’échantillonnage, son extraction et sa normalisation documentées. On ne lui imposera pas aveuglément le pipeline du ViT image.

## 3. Travaux de référence et conséquences pour le projet

### 3.1 Descripteurs et validité de l’évaluation

**Tzanetakis et Cook, 2002 — Musical Genre Classification of Audio Signals.** Le travail organise les caractéristiques autour du timbre, du rythme et des hauteurs pour classer les genres. Il justifie une référence interprétable utilisant plusieurs familles de descripteurs. Il ne fournit pas une règle universelle reliant un descripteur à un genre ; ses résultats historiques ne sont pas transposables directement à FMA. [Article des auteurs](https://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf)

**Sturm, 2013 — Classification accuracy is not enough.** L’auteur montre pourquoi un score agrégé ne suffit pas à caractériser le comportement musical d’un système. Conséquence : écouter les erreurs, examiner les raccourcis possibles et tester des perturbations contrôlées. Nous ne qualifierons pas le modèle de « compréhension musicale » sur la seule base d’une accuracy. [Article](https://link.springer.com/article/10.1007/s10844-013-0250-y)

**Sturm, 2014 — The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use.** L’analyse documente répétitions, erreurs d’étiquettes et distorsions, qui affectent l’interprétation des comparaisons. Elle motive notre audit des données ; elle ne signifie pas que FMA serait exempt de problèmes. [Publication et résumé institutionnels](https://vbn.aau.dk/en/publications/the-gtzan-dataset-its-contents-its-faults-their-effects-on-evalua/)

**Defferrard et al., 2017 — FMA: A Dataset for Music Analysis.** FMA fournit audio, métadonnées et partitions conçues notamment pour limiter l’effet artiste. Nous partons de FMA-small et vérifions ces propriétés après nettoyage. Ses catégories reflètent sa collection et ses annotations ; les résultats n’établiront pas une couverture de toute la musique commerciale ou du gospel. [Article](https://arxiv.org/abs/1612.01840) · [Données et code](https://github.com/mdeff/fma)

### 3.2 De l’image au spectrogramme

**Dosovitskiy et al., ICLR 2021 — An Image is Worth 16×16 Words.** ViT traite une image comme une séquence de patches avec information de position. Le transfert après préentraînement est central dans les expériences présentées. Conséquence : commencer par un modèle préentraîné et comprendre la taille des patches, les positions et l’agrégation des tokens. L’article porte sur des images ; son succès ne prouve pas celui du transfert musical. [Article](https://arxiv.org/abs/2010.11929)

**Touvron et al., ICML 2021 — DeiT.** Le travail étudie un entraînement plus efficace en données et la distillation avec un enseignant. Cela documente l’importance de la recette d’apprentissage, au-delà de l’architecture. « Data-efficient » ne veut pas dire qu’un entraînement complet sur un ordinateur personnel ARM est automatiquement réaliste. [Article et actes](https://proceedings.mlr.press/v139/touvron21a.html)

**Gong, Chung et Glass, Interspeech 2021 — AST: Audio Spectrogram Transformer.** AST adapte le Transformer à des patches de spectrogrammes et propose des poids préentraînés. C’est le comparateur naturel du ViT image. Son prétraitement spécifique doit être respecté. Ses résultats sur AudioSet ou ESC-50 ne sont pas des résultats de classification des huit genres FMA. [Article](https://arxiv.org/abs/2104.01778) · [Code officiel](https://github.com/YuanGongND/ast)

**Koutini et al., Interspeech 2022 — PaSST.** Patchout retire des patches pendant l’entraînement pour réduire le coût et régulariser le modèle. Ce travail rend explicite le compromis entre séquence de tokens et calcul. Nous le retenons comme piste d’efficacité, sans reproduire son préentraînement complet. Retirer des tokens n’est pas équivalent à simplement masquer les valeurs du spectrogramme. [Article](https://arxiv.org/abs/2110.05069) · [Code](https://github.com/kkoutini/PaSST)

**Kong et al., 2020 — PANNs.** Ces CNN préentraînés sur AudioSet servent de représentations transférables pour plusieurs tâches audio. Ils rappellent qu’une comparaison pertinente ne se limite pas à un Transformer et une référence très faible. Un PANN compact pourra compléter notre petit CNN si le budget le permet. Architecture, taille et préentraînement devront être distingués dans l’interprétation. [Article](https://arxiv.org/abs/1912.10211)

### 3.3 Représentations spécialisées pour la musique

**Alonso-Jiménez, Serra et Bogdanov, 2023 — MAEST.** Les auteurs évaluent l’initialisation, la durée des segments, les représentations de différents blocs et le coût d’extraction. Dans leurs expériences, des couches intermédiaires peuvent mieux transférer que la dernière et des contextes plus longs sont utiles. Conséquence : comparer deux couches fixées à l’avance et plusieurs fenêtres. Ce sont des résultats situés, pas une garantie sur notre ViT-Tiny. [Article et méthode](https://arxiv.org/html/2309.16418v1)

**Li et al., prépublication 2023 / ICLR 2024 — MERT.** MERT combine des cibles acoustiques avec une cible CQT introduisant des informations de hauteur et d’harmonie pendant le préentraînement. Cela motive l’étude de la complémentarité entre caractéristiques apprises et musicales. MERT reçoit du signal audio et n’est pas simplement un ViT sur des images log-Mel. Nous ne reproduirons pas son préentraînement de grande taille. [Article](https://arxiv.org/abs/2306.00107)

**Won, Hung et Le, ICASSP 2024 — MusicFM.** L’étude examine notamment la représentation, la résolution temporelle et le transfert entre tâches musicales. Son dépôt fournit une version préentraînée sur FMA-large : la comparer naïvement sur FMA-small exposerait le test à des données déjà vues sans labels. Cette version sera exclue du classement principal ; l’exposition au préentraînement sera documentée pour chaque modèle. [Article](https://arxiv.org/abs/2311.03318) · [Dépôt officiel](https://github.com/minzwon/musicfm)

**Zhu et al., 2025 — MuQ.** MuQ apprend à prédire des cibles discrètes issues d’une quantification résiduelle de représentations Mel. Il illustre une direction récente d’apprentissage auto-supervisé musical. À retenir comme comparaison bibliographique, puis éventuellement comme extracteur gelé si les ressources le permettent. Ses résultats publiés ne suffisent pas à le proclamer meilleur sur notre protocole. [Article](https://arxiv.org/abs/2501.01108)

### 3.4 Augmentation des données

**Park et al., Interspeech 2019 — SpecAugment.** Le travail initial porte sur la parole et propose notamment des masquages temporels et fréquentiels. Leur intérêt pour nos genres devra être mesuré. Pas de retournements ou rotations copiés automatiquement de la vision ; tempo et transposition peuvent modifier des indices utiles. [Article](https://arxiv.org/abs/1904.08779)

### 3.5 Séparation harmonique/percussive : complément du 21 octobre 2026

**FitzGerald, DAFx 2010 — Harmonic/Percussive Separation Using Median Filtering.**
Les structures durables et les événements brefs présentent des orientations
différentes dans un spectrogramme. Des filtres médians selon le temps et selon
la fréquence produisent deux estimations, utilisées pour construire des masques
appliqués au mélange. La méthode motive notre représentation à trois canaux ;
l’article ne démontre pas un gain de classification de genres avec un ViT.
Consultation : résumé, introduction et méthode de filtrage médian.
[Article des actes DAFx](https://dafx.de/paper-archive/2010/DAFx10/DerryFitzGerald_DAFx10_P15.pdf)

**Driedger, Müller et Disch, ISMIR 2014 — Extending Harmonic-Percussive Separation
of Audio Signals.** Les auteurs examinent les fuites entre composantes et les
sons qui ne sont clairement ni harmoniques ni percussifs. Ils introduisent un
facteur de séparation et une composante résiduelle. Cela motive une interprétation
prudente des canaux, plutôt que de les assimiler à des instruments isolés.
Consultation : résumé, introduction et section sur le facteur de séparation.
[Article des auteurs](https://www.audiolabs-erlangen.de/fau/professor/mueller/publications/2014_DriedgerMuellerDisch_ExtensionsHPSeparation_ISMIR.pdf)

**Décision expérimentale du projet.** Employer l’implémentation librosa avec
filtres 31/31, masques doux de puissance 2 et marges 1/1, fixés avant les scores.
La séparation précède les filtres Mel ; l’ordre des entrées est original/H/P.
Les trois canaux conservent une référence dB commune. Cette convention ne
reproduit pas toutes les variantes des articles et ne prétend pas optimiser
la séparation. Voir le [protocole HPSS](archive/ABLATION_HPSS.md) et la
[documentation de l’implémentation](https://librosa.org/doc/0.11.0/generated/librosa.decompose.hpss.html).

## 4. Ce que nous proposons d’étudier

Notre contribution sera expérimentale et reproductible. Il ne s’agit pas de revendiquer une nouvelle architecture ou une fusion inédite sans recherche supplémentaire.

| Question | Comparaison proposée | Conclusion autorisée |
|---|---|---|
| Le rythme et l’harmonie complètent-ils le ViT ? | Descripteurs seuls / ViT seul / fusion des deux | Complémentarité mesurée sur ce dataset et ce protocole |
| La représentation d’entrée compte-t-elle ? | Log-Mel répété / canaux original + harmonique + percussif | Effet du changement de représentation, incluant son décalage avec le préentraînement |
| Quels passages faut-il observer ? | Une fenêtre centrale / trois fenêtres de même durée | Effet de la couverture temporelle, sans prétendre isoler la durée de contexte du Transformer |
| Quelle couche transférer ? | Tokens du bloc 6 / du bloc 12 du même ViT | Effet du niveau de représentation, à entrée et classificateur identiques |
| Quel coût pour quel gain ? | Macro-F1, mémoire, extraction et inférence | Compromis mesuré sur le matériel disponible |

La comparaison ViT image / AST audio restera une extension prioritaire : elle change plusieurs facteurs à la fois et ne démontrera pas, seule, l’effet causal du préentraînement audio. MERT et MuQ restent d’abord des références, pour éviter d’accumuler les grands modèles.

## 5. Protocole scientifique

- Garder les mêmes morceaux, partitions et fenêtres pour les comparaisons contrôlées. Aucun découpage aléatoire des fenêtres entre train et test.
- N’ajuster normalisations, sélection de caractéristiques, hyperparamètres et éventuelle calibration qu’avec train/validation. Recalculer le cache lorsque modèle, fenêtre ou prétraitement change.
- Séparer évaluations par morceau et par fenêtre ; la mesure principale est par morceau.
- Rapporter macro-F1, accuracy, F1 par genre, matrice de confusion et coût. Estimer les intervalles par bootstrap groupé par artiste, pas en considérant toutes les fenêtres comme indépendantes.
- Répéter les entraînements stochastiques finaux avec trois graines ; ne pas prétendre que répéter un classificateur déterministe mesure une variabilité d’entraînement.
- Réserver le test aux configurations finalisées. Les analyses exploratoires d’erreurs auront lieu sur validation ; les erreurs de test pourront être décrites après verrouillage, sans réoptimiser dessus.
- Examiner la robustesse à un changement modéré de gain sans écrêtage et à un bruit faible. Tester une tendance attendue, sans promettre l’invariance parfaite.
- Une carte d’attention n’est pas une preuve causale. Utiliser en complément le masquage de régions temps/fréquence et interpréter prudemment ces perturbations potentiellement hors distribution.
- Documenter les données de préentraînement connues et les recouvrements possibles. L’absence de métadonnées ne permet pas de certifier l’absence de contamination.

## 6. Ordre de lecture et fiches

Parcours prioritaire : Tzanetakis–Cook et FMP → Sturm et FMA → ViT → AST → MAEST. Parcours complémentaire : DeiT, PaSST, PANNs, MERT, MusicFM, MuQ et SpecAugment.

Chaque fiche de lecture approfondie devra contenir : référence et version ; question ; données et partitions ; représentation et paramètres audio ; architecture et préentraînement ; protocole ; résultat utile ; limites ; décision pour notre projet ; expérience reproductible associée. Les affirmations tirées des articles resteront distinctes de nos hypothèses.

Le [plan révisé](archive/PLAN_PROJET.md) traduit cette revue en jalons. Les propositions
de la revue restent des hypothèses : les expériences réalisées depuis sa rédaction
et leurs résultats vérifiés sont référencés dans [l’index des résumés](resumes/INDEX.md).
