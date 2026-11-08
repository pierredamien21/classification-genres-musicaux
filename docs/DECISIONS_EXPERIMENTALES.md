# Décisions et pistes expérimentales

Mis à jour : 28 octobre 2026. Ce registre distingue résultats vérifiés, décisions
de travail et hypothèses non testées. Les scores ci-dessous concernent uniquement
la validation ; le jeu de test ne guide pas ces choix.

## Décisions issues des expériences

| Observation | Décision ou conséquence | Preuve |
|---|---|---|
| Trois fenêtres : macro-F1 0,5447 ; centrale seule : 0,4882 au bloc 12 | Conserver trois fenêtres dans la référence ; la comparaison change aussi le nombre d’exemples de train | [Fenêtres](../reports/ABLATION_FENETRES.md) |
| Bloc 6 : 0,5780 ; bloc 12 : 0,5447, à canaux identiques | La profondeur intermédiaire mérite d’être conservée comme candidate ; résultat lié à la convention de LayerNorm documentée | [Blocs](../reports/ABLATION_BLOCS.md) |
| Original/H/P au bloc 12 : 0,5844 ; original répété : 0,5447 | Gain observé à profondeur fixée ; référence conservée pour la combinaison | [HPSS](../reports/ABLATION_HPSS.md) |
| Bloc 6 + HPSS : 0,5975, accuracy 0,6128 ; +0,0131 face à bloc 12 HPSS et +0,0195 face à bloc 6 original | Candidat principal pour préparer l’évaluation finale ; aucune supériorité garantie sur test ou sur chaque genre | [Combinaison](../reports/COMBINAISON_BLOC6_HPSS.md) |
| Face à bloc 12 HPSS : 56 corrections et 44 régressions, dont 9 régressions chez un même artiste ; Pop 13/94 corrects, Instrumental 8 corrections/12 régressions | Rapporter précision/rappel et résultats par genre ; conserver le bootstrap groupé par artiste prévu pour le test. Ne pas attribuer de cause acoustique sans écoute | [Régressions](../reports/REGRESSIONS_BLOC6_HPSS.md) |
| Fusion complète : 0,5219, derrière le ViT original à 0,5447 | Ne pas présenter la fusion comme un gain ; toute nouvelle fusion exige une hypothèse différente et une comparaison explicite | [Fusion](../reports/COMPARAISON_FUSION.md) |
| CNN 42/43/44 à dix époques : macro-F1 moyenne 0,5353 ± 0,0103 ; époques retenues 9/10/8 | Répétitions conservées comme comparateur initial ; prolongation contrôlée réalisée | [Répétitions CNN](../reports/REPETITIONS_CNN.md) |
| Plafond vingt : moyenne 0,5407 ± 0,0105 ; gain +0,0054 porté par la seule graine 42 ; arrêts à 18/13/11 | Conserver les deux budgets ; ne pas prolonger encore sans nouvelle hypothèse. Priorité aux erreurs par genre ; pas de conclusion sur la capacité maximale du CNN | [Extension CNN](../reports/EXTENSION_CNN.md) |
| Sur 780 morceaux, la moyenne dépasse chaque position seule pour les trois ViT ; HPSS : 73/314 erreurs avec une fenêtre correcte, Pop : 10/82 | Conserver les trois fenêtres et la moyenne ; pas de recherche de poids temporels justifiée à ce stade. Les désaccords ne prouvent pas leur cause | [Temporalité](../reports/ANALYSE_TEMPORELLE.md) |

## Pistes à examiner avant le verrouillage

L’analyse historique des trois ViT antérieurs trouvait 71/94 morceaux Pop manqués
en commun. La nouvelle comparaison concerne un trio différent (bloc 6 HPSS,
bloc 12 HPSS, bloc 6 original) : 74/94 morceaux Pop restent manqués par les trois.
Ces comptes ne mesurent donc pas une évolution sur un ensemble de modèles identique.
Dix-neuf fiches d’écoute historiques sont préparées, avec dix-neuf artistes distincts ;
les observations humaines restent à recueillir.

La combinaison est terminée. Pour Pop, la F1 passe de 0,1633 à 0,1884 face au
bloc 12 HPSS : seulement un vrai positif net supplémentaire (12 → 13), mais
dix faux positifs de moins (41 → 31). Hip-Hop gagne six morceaux sans régression
face à cette référence ; il reste derrière bloc 6 original en F1 (0,7333 contre
0,7610). Instrumental perd quatre vrais positifs nets et gagne sept faux positifs.
Ces résultats ne suffisent pas à identifier une cause ni à choisir un nouveau
prétraitement. Les labels sont conservés sans correction automatique.

**Décision de travail :** conserver bloc 6 + HPSS comme candidat principal et
préparer les contrôles de robustesse et le protocole final. Ce n’est pas encore
un verrouillage ni une ouverture du test. Aucune nouvelle grille, combinaison
par genre ou pondération temporelle n’est lancée. La question principale a une
réponse négative dans le protocole testé : la fusion des descripteurs avec le
bloc 12 n’a pas amélioré la référence. HPSS apporte un résultat positif distinct,
qui ne prouve pas que le modèle comprend mieux l’harmonie.

| Priorité de revue | Observation et hypothèse | Expérience ou analyse envisagée | Coût et critère de décision |
|---|---|---|---|
| Complément — Revue d’écoute | Confusions et désaccords mesurés ; 19 extraits sélectionnés avec contrôles corrects et diversité des artistes | Recueillir des observations avant de révéler les labels, puis comparer aux métadonnées ; aucun réétiquetage automatique | Fiches préparées, audio original vérifié ; page et premier son confirmés par l’utilisateur, export non vérifié. Échantillon ciblé, non représentatif ; ne pas attribuer de cause sans observation |
| Prochaine étape — Préparation finale | Meilleur score global, faiblesses Pop/Instrumental et dépendance entre morceaux d’un artiste | Définir robustesse, règles de sélection et protocole de test ; préparer la liste des artefacts et comparateurs figés | Train/validation pour les vérifications préalables ; test fermé jusqu’au protocole final. Pas d’intervalle de généralisation revendiqué sur la sélection actuelle |
| Facultatif — Transfert spécialisé | Le ViT utilise un préentraînement image et un redimensionnement carré | AST ou affinage limité, uniquement si une question supplémentaire justifie l’extension | Non lancé. Vérifier prétraitement, poids, exposition aux données et coût avant adoption ; plusieurs facteurs changent. Peut être reporté à une version suivante |

## Règle d’évolution

### Reproduction des trois CNN — T5, 29 octobre 2026

Les graines 42/43/44, aux époques déjà sélectionnées 15/10/8, reproduisent les
64 fichiers du panel et les métriques complètes des 780 morceaux depuis le cache
natif. Écart mesuré nul, y compris par fenêtre, malgré des lots d’au plus trois
fenêtres en inférence fichier contre seize en validation complète.

Décision : T5 et le point B satisfaits ; conserver les trois exports et leur
moyenne de métriques, sans nouvelle sélection ni ensemble. Les fonctions de
prétraitement et de calcul CNN sont réutilisées, sans chemin ViT substitué.
T6 doit désormais vérifier toutes les preuves et créer un manifeste de verrouillage
distinct. Aucun test final lancé et aucun entraînement supplémentaire.
[Rapport et preuves](../reports/INFERENCE_CNN.md).

### Reproduction des comparateurs linéaires — T4, 28 octobre 2026

Les six systèmes linéaires reproduisent les probabilités des 64 morceaux du
panel et les métriques des 780 morceaux depuis les caches. Le contrôle révèle
deux différences d’exécution : arrondi des durées de padding et nombre de threads
BLAS. Comparer la géométrie en échantillons et rétablir la limite historique d’un
thread résout ces écarts ; la tolérance reste 10⁻⁶. Les erreurs finales sont
inférieures à 9 × 10⁻¹⁶ par fenêtre. Tests de régression conservés.

Décision : T4 satisfaite, poursuivre T5 sur les trois CNN déjà sélectionnés.
La fusion reçoit une fenêtre commune aux deux extracteurs et vérifie ses colonnes
contre l’export. Aucun poids, scaler ni prétraitement acoustique modifié ; sources
et preuves historiques préservées. Les limites scientifiques et le test fermé
restent inchangés. [Rapport](../reports/INFERENCE_COMPARATEURS.md).

### Incertitude sur validation — T3, 28 octobre 2026

Le bootstrap prévu est vérifié sur les 780 morceaux et 219 artistes de validation.
2 000 tirages communs aux neuf exports, majoritaire et moyenne CNN ; aucun audio
chargé. Écart candidat − ViT original en macro-F1 : +0,0528, intervalle percentile
95 % [+0,0257 ; +0,0785]. Les intervalles avec bloc 12 HPSS et bloc 6 original
contiennent zéro en macro-F1, comme celui fusion − ViT. Aucun gain des variantes
proches n’est déclaré établi sur des données indépendantes.

Décision : conserver les modèles et comparaisons annoncés, poursuivre T4 puis T5.
Ces intervalles conditionnent sur des modèles sélectionnés avec cette validation ;
ils ne corrigent pas le biais de sélection ni la multiplicité des comparaisons.
Ils ne mesurent pas toute la variabilité d’entraînement. Aucun ajustement du nombre
de tirages ou de graine après résultat. [Rapport et preuves](../reports/INCERTITUDE_VALIDATION.md).

### Comparaison T2b et limite acceptée — 28 octobre 2026

T2b reproduit les 64 références propres du bloc 6 original à 4,47 × 10⁻⁷,
puis applique les mêmes perturbations que T2. Sous bruit : perte macro-F1
0,1012 pour l’original contre 0,1831 pour HPSS, mais 28 changements contre 26.
Les macro-F1 bruitées sont proches (0,3681 et 0,3648) ; HPSS conserve une
accuracy supérieure (26/64 contre 24/64). Les deux critères annoncés divergent :
aucun gagnant global en robustesse, aucune causalité attribuable aux seuls canaux.

Décision pour cette version : conserver le candidat HPSS sélectionné sur la
validation complète et accepter explicitement la sensibilité au bruit comme
limite mesurée. Les déclencheurs ont été examinés, pas supprimés. Aucun
débruitage, normalisation nouvelle, entraînement ou seuil modifié. Une extension
corrective reste possible avec un protocole préalable ; elle n’est pas lancée.
Le point A est satisfait ; poursuivre T3, puis les autres comparateurs avant
verrouillage. Test toujours fermé. [Rapport et preuve](../reports/COMPARAISON_ROBUSTESSE_BLOC6.md).

### Diagnostic T2a et comparaison T2b — 28 octobre 2026

T2a reproduit exactement 72 fenêtres des huit pilotes. Pas d’augmentation
massive des valeurs hors bornes avant redimensionnement ; le déplacement L2
standardisé moyen est 6,82 fois plus élevé sous bruit que sous gain. Les marges
initiales sont plus faibles en médiane parmi les 26 changements du panel, mais
certaines sont élevées. Les deux changements sous bruit des huit pilotes restent
des erreurs dans les deux conditions ; ils ne reproduisent pas les dix régressions
du panel complet. Aucune cause unique ou panne de code établie.

Décision : **T2b, comparer au bloc 6 original existant**, mêmes 64 morceaux,
conditions, graines et fenêtres ; identité vérifiée d’abord, pilote huit morceaux,
budget trente minutes. Rapporter pertes de F1, changements et compromis avec
les scores propres ; aucun réentraînement ni sélection sur test. Le rôle spécifique
du système HPSS reste une question, pas une conclusion. Une correction éventuelle
sera définie séparément. [Mesures](../reports/DIAGNOSTIC_BRUIT.md) et
[critères T2b](../tasks/todo.md).

### Revue de sensibilité — T2, 28 octobre 2026

T2 est terminée sur le panel fixé : macro-F1 identité 0,5479, gain −6 dB 0,5050,
bruit SNR 20 dB 0,3648. Sous bruit, 26/64 classes changent et la perte de F1
atteint 0,1831 : les deux déclencheurs prévus sont franchis. Gain : six changements
et perte 0,0429, sous les seuils. Les contrôles techniques et le recalcul indépendant
confirment la mesure ; ils ne démontrent pas la robustesse du classificateur.

Décision : ajouter **T2a avant verrouillage**, sur les huit morceaux pilotes
déjà choisis, pour examiner canaux original/H/P, valeurs bornées, distances
d’embeddings et marges. Ces mécanismes sont des hypothèses, sans cause encore
établie. Budget dix minutes de calcul, pas d’entraînement, pas de changement
des seuils ni de sélection supplémentaire selon les erreurs. Le diagnostic
déterminera si une extension corrective est justifiée ou si la sensibilité
reste une limite explicite. [Résultats et analyse](ROBUSTESSE_CANDIDAT.md).

### Préparation finale du 28 octobre 2026

Le [protocole préparatoire](PROTOCOLE_EVALUATION_FINALE.md) recense neuf exports
vérifiés pour sept systèmes appris, plus le majoritaire. Ils restent entraînés
sur train seul : pas de réentraînement train + validation ni de nouvelle
sélection d’époque CNN. Ce choix facilite la reproduction exacte des modèles
déjà comparés, au prix possible d’un gain lié à davantage de données.

Le panel de sensibilité est fixé avant ses résultats : 64 morceaux de validation,
huit par genre et un par artiste, sélection sans prédictions. Les essais sont
terminés et déclenchent T2a ; le bootstrap reste à réaliser. L’inférence commune
du candidat ([T1](../tasks/todo.md)) est vérifiée sur 64 morceaux/192 fenêtres
à 10⁻⁶. Le recensement n’est pas
un verrouillage et le test reste désactivé dans la configuration préparatoire.

### Politique maintenue

Avant une extension, inscrire la question, la configuration, les comparateurs,
les métriques et le budget. Réviser explicitement le plan si l’expérience dépasse
son enveloppe initiale ; conserver les résultats négatifs et les essais antérieurs.
La seule différence de quelques millièmes sur validation ne suffit pas à prouver
une amélioration généralisable. Les critères de coût, simplicité et reproductibilité
peuvent aussi intervenir, en étant annoncés.

Après ce point de décision et les extensions retenues, figer une version pour
son évaluation finale. Le projet peut évoluer ensuite, mais une version optimisée
sur les erreurs de ce test ne pourra plus revendiquer ce même test comme une
mesure indépendante de sa sélection.

## Verrouillage T6 — 29 octobre 2026

Les preuves T1–T5 et les revues T2a/T2b sont vérifiées, avec 7 985 artefacts
épinglés et 231 tests réussis. Un nouveau manifeste verrouille les neuf exports
et les paramètres préparés, sans réécrire les preuves historiques. Sensibilité
au bruit et absence de revue subjective sont conservées comme limites.
Le coût d’inférence test est projeté à 95,3 minutes depuis huit morceaux train ;
marge de planification : 1 h 35–3 h 10, hors contrôles, bootstrap et rapports.
T7 reste à implémenter et à vérifier sur validation avant le test. Aucun score
test n’a été consulté. [Preuves et procédure](../reports/VERROUILLAGE_FINAL.md).

## 29 octobre — clôture de l’évaluation V1

Test et contrôle indépendant terminés. Candidat bloc 6 HPSS : macro-F1 0,5035,
contre 0,5975 sur validation ; la cause de cette baisse n’est pas établie.
Gain principal sur ViT original +0,0394, IC apparié [+0,0054 ; +0,0715].
Les trois comparaisons secondaires incluent zéro. La fusion passe légèrement
devant le ViT original sur test sans gain établi : conserver ce résultat
ainsi que sa baisse sur validation, sans changer le candidat.

Décision : conserver exports et protocole V1 ; passer aux livrables pédagogiques
et à la démo. Aucun ajustement supplémentaire motivé par le test. La session
autonome utilisateur reste prévue avant le bilan général comme développement
complémentaire. [Preuves et limites](../reports/EVALUATION_FINALE.md).
