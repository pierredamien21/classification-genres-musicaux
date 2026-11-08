# Session d’entraînement autonome avant la conclusion générale

**7 novembre : session complète terminée et vérifiée.** Trente époques, meilleure 26, macro-F1 validation 0,5707 ; [résultats](../reports/SESSION_UTILISATEUR.md) et [conclusion](CONCLUSION_PROJET.md). La préparation datée ci-dessous est historique. Aucun lancement supplémentaire nécessaire.

Mise à jour du 5 novembre 2026 : **préparation réalisée, entraînement complet non lancé**.
Le notebook `05_entrainement_utilisateur.ipynb` et `.venv-notebook` sont disponibles.
Recette fixée : CNN depuis zéro, graine 42, plafond 30 époques, patience 5,
amélioration strictement positive (seuil 0), égalité en faveur de la première
époque, AdamW 0,001 et budget de 7 200 secondes. Pas d’augmentation ni accès test.
Le petit essai couvre 16 morceaux train et 16 validation, deux époques avec
pause puis reprise en arrière-plan ; métriques recomptées et export vérifiés.
Les sources V1 sont conservées ; l’orchestration utilisateur est distincte.

Estimation au plafond : 35–80 minutes selon la charge, marge calculée à 74 minutes
sur les durées historiques. Ce n’est pas une durée garantie. Environnement créé
par copies physiques des paquets locaux ; installation réseau neuve non testée.

[Commandes et mode d’emploi](GUIDE_LIVRAISON.md). Preuves :
`reports/USER_SESSION_READY.json`, `reports/USER_SMOKE_JOB.json`.
Le notebook a été exécuté avec `LANCER=False`. L’utilisateur lance puis signale
la fin ; les résultats complets seront vérifiés avant la conclusion générale.

**Les sections suivantes conservent les notes de conception du 29 octobre.**
Les choix ci-dessus remplacent les options qui y étaient encore ouvertes.

## But et place dans le projet

Permettre à l’utilisateur de lancer lui-même une expérience depuis un notebook,
de suivre l’apprentissage et de revenir une fois les calculs terminés. L’assistant
relira alors les sorties sauvegardées, vérifiera leur provenance et intégrera
les résultats, y compris un éventuel échec ou une absence d’amélioration.

Cette session complète la campagne V1 verrouillée ; elle ne remplace ni ses
exports ni ses résultats. La préparation et l’exécution utilisateur devront être
traitées avant de déclarer le projet et sa conclusion générale terminés.

## Ce qui a déjà été entraîné

Les CNN ont d’abord eu un plafond de dix époques, puis une prolongation jusqu’à
vingt sur les trois graines 42/43/44. Avec une patience de trois, les arrêts ont
eu lieu aux époques 18/13/11 ; les meilleurs checkpoints sont 15/10/8. Vingt était
un maximum, pas une obligation. Voir [la synthèse vérifiée](../reports/EXTENSION_CNN.md).

Le ViT actuel est gelé et ses embeddings alimentent une régression logistique.
Une demande de « trente époques de ViT » correspondrait à un nouvel affinage,
avec paramètres entraînables, optimiseur et budget propres. Cet affinage n’est
pas inclus implicitement dans une simple prolongation du CNN.

## Protocole à fixer avant le lancement

Point de départ proposé : une expérience CNN avec **30 époques au maximum**,
sélection par macro-F1 de validation par morceau et restauration du meilleur
checkpoint. La session peut aussi reproduire le plafond vingt pour apprendre
à utiliser les outils. Aucun score meilleur n’est promis.

La patience doit être explicite, de l’ordre de **3 à 5 époques sans amélioration**.
Conserver trois reproduit la règle historique ; passer à cinq teste une règle
d’arrêt différente et doit être déclaré comme tel. Un plafond trente avec la
même patience ne change pas un entraînement déjà arrêté avant vingt. Ne pas
remettre à zéro un compteur de patience épuisé en prétendant reprendre à l’identique.

Avant exécution, figer dans une nouvelle configuration : modèle, plafond, patience,
seuil minimal d’amélioration, règle d’égalité, données, graines, optimiseur, taux
d’apprentissage et budget de temps. Le seuil ne sera pas choisi après lecture
des courbes. Distinguer le critère de progression de l’arrêt anticipé de la règle
de sélection des meilleurs poids, et documenter les deux.

Deux modalités possibles, à choisir à la préparation : réentraînement depuis
une initialisation définie, ou continuation d’un checkpoint complet explicitement
identifié. Une continuation fidèle exige poids courants, optimiseur, état aléatoire,
historique et état de l’arrêt anticipé. Une nouvelle patience constitue une nouvelle
recette. Comparer les mêmes graines lorsque le budget le permet ; un seul run
pédagogique ne suffit pas à établir une supériorité sur la moyenne de trois graines.

Faire un court essai de fonctionnement et chronométrer avant de fournir la durée
estimée. Conserver une limite de temps, des sauvegardes et une reprise vérifiée.
Un arrêt par budget doit être distingué d’un arrêt par patience ou par plafond.

## Environnement et notebook à préparer

Prévoir un environnement isolé, par exemple `.venv-notebook`, avec versions
explicites, installation du package local et noyau Jupyter identifiable. Vérifier
les imports, le décodage audio, le chargement des données et un petit entraînement.
Le notebook affichera son interpréteur et le matériel détecté. Sur la machine
actuelle, CPU validé ; ne pas présumer de CUDA ni d’une accélération Adreno.

Support proposé : `05_entrainement_utilisateur.ipynb`, séparé du parcours de
lecture rapide. Il montrera configuration, durée estimée, lancement volontaire,
progression, courbes, meilleur checkpoint et emplacement des résultats. Réutiliser
les fonctions du package plutôt que recopier une boucle d’apprentissage. Les
modifications nécessaires seront réalisées plus tard, hors des sources figées
de la campagne en cours ou dans une version distincte tracée.

Le notebook doit conserver les résultats sur disque pendant l’exécution, même si
le navigateur se ferme. Aucun lancement de plusieurs heures par inadvertance
depuis le notebook principal. Donner les commandes de lancement et de reprise
et les indications pour maintenir la machine disponible pendant le calcul.

## Sorties attendues pour la reprise par l’assistant

Un nouveau dossier d’expérience contiendra : configuration résolue, versions et
interpréteur, empreintes du code/données/parents, graines, matériel, journal,
historique par époque (pertes, métriques train/validation, durées et taux
d’apprentissage), dernier checkpoint complet, meilleurs poids et prédictions de
validation par morceau. Ajouter époques réalisées/retenues, motif d’arrêt et état
final explicite : terminé, interrompu ou échoué. Prévoir aussi un export léger
des courbes et du notebook exécuté, sans dépendre de ses seules sorties visuelles.

Lorsque l’utilisateur annonce la fin, vérifier d’abord le statut et la complétude,
puis reconstruire les métriques depuis les prédictions, contrôler les courbes et
comparer aux références compatibles. Aucun run inachevé n’est annoncé comme fini.
Après vérification, intégrer les résultats au notebook principal et mettre à jour
les conclusions et limites avant le bilan général.

## Place du test et portée des conclusions

La nouvelle expérience utilise train pour apprendre et validation pour choisir
les checkpoints. Elle ne consulte pas le test à chaque époque. Ne pas choisir
une patience, un modèle ou une graine selon les performances test.

Si le test V1 a déjà été consulté à ce moment-là, présenter cette extension comme
un développement complémentaire. Une réutilisation éventuelle de ce test doit
être déclarée et ne fournit pas une nouvelle évaluation indépendante après
adaptation. Pour revendiquer une amélioration indépendante, il faudra un protocole
supplémentaire et des données tenues à l’écart. Les conclusions V1 demeurent
attachées à leurs modèles verrouillés.

## Ordre retenu et conditions de clôture

1. Achever les livrables prévus et conserver les preuves de la campagne V1.
2. Préparer et vérifier l’environnement et le notebook d’entraînement autonome.
3. Fixer la recette complémentaire et le budget avant tout lancement.
4. L’utilisateur lance, puis indique la fin ; les fichiers assurent la continuité.
5. L’assistant vérifie et exploite les sorties ; le notebook distingue les deux campagnes.
6. Rédiger ensuite la conclusion générale, avec coûts, limites et résultats négatifs.

L’existence d’un meilleur score n’est pas une condition de réussite de cette
session. Son intérêt est aussi pédagogique : exécution autonome, reprise,
traçabilité et interprétation correcte d’un budget d’apprentissage.
