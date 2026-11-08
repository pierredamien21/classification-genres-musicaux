# Préparation de l’évaluation finale — version 1

Préparation : 28 octobre 2026 ; suivi mis à jour le 29 octobre. **Protocole final verrouillé ; campagne test et contrôles indépendants terminés.** [Résultats V1](../reports/EVALUATION_FINALE.md).
Avancement : T1 est vérifiée depuis cette préparation, sur 64 morceaux et
192 fenêtres (écart maximal 5,18 × 10⁻⁷). Voir [l’inférence](INFERENCE_CANDIDAT.md).
T2 est également exécutée : le bruit provoque une perte de macro-F1 de 0,1831
et 26/64 changements, dépassant les deux seuils de revue. T2a/T2b sont terminées : l’original perd moins de F1 mais change plus de classes. Aucun gagnant global ; HPSS conservé avec sensibilité acceptée comme limite. T3 est vérifiée sur validation ; T4 reproduit les six systèmes linéaires. T5 est vérifiée depuis le 29 octobre : les trois CNN reproduisent le panel et la validation complète. T6 est terminée : 7 985 artefacts vérifiés, manifeste publié et coût mesuré sur train. T7 a vérifié son évaluateur sur les 780 morceaux de validation, avec recomptage indépendant ; la campagne test et son contrôle indépendant sont terminés. Voir [le bilan](resumes/37_EVALUATION_FINALE.md). Voir [le verrouillage](VERROUILLAGE_EVALUATION.md) et [son rapport](../reports/VERROUILLAGE_FINAL.md).
Voir le [diagnostic](../reports/DIAGNOSTIC_BRUIT.md).
Le [bootstrap](../reports/INCERTITUDE_VALIDATION.md) est implémenté et vérifié : 2 000 tirages sur 219 artistes, 261 tests réussis dans la suite actuelle et recalcul indépendant exact. Les résultats restent conditionnels aux modèles sélectionnés sur cette validation.
Les paramètres machine sont dans `configs/final_evaluation_protocol.json`.
Cette étape formalise la décision de la partie 22. Elle n’annonce ni une
robustesse acquise, ni une estimation finale des performances.

## Question et modèles retenus

Évaluer la généralisation du candidat sélectionné sur validation, bloc 6 + HPSS,
et conserver les comparaisons nécessaires pour raconter l’expérience complète.

| Rôle | Système | Artefact retenu |
|---|---|---|
| Candidat principal | Bloc 6 + HPSS | C = 0,1 déjà ajusté sur train |
| Référence principale | ViT bloc 12 original | Export existant |
| Hypothèse de complémentarité | Descripteurs seuls et fusion bloc 12 + descripteurs | Deux exports existants |
| Effets profondeur/canaux | Bloc 6 original et bloc 12 HPSS | Deux exports existants |
| Référence apprise de zéro | CNN plafond vingt, graines 42/43/44 | Trois checkpoints sélectionnés sur validation |
| Repère trivial | Classe majoritaire de train | Pas d’ajustement sur test |

Cela représente neuf exports pour sept systèmes appris, plus le majoritaire.
Les ablations centrale seule, fusion sans rythme/harmonie et CNN plafond dix
restent dans les résultats de développement ; elles ne seront pas ajoutées au
test en fonction des premiers résultats. Les chemins et empreintes des exports
sont recensés dans `reports/PREPARATION_EVALUATION_FINALE.json`.

**Aucun réentraînement sur train + validation.** Cette décision conserve les
artefacts effectivement comparés, évite une nouvelle sélection d’époque CNN
et simplifie la vérification entre validation, test et future démo. Elle peut
coûter de la performance face à un réentraînement avec davantage de données.
Les paramètres et normalisations restent ceux appris sur train.

## Chaîne à vérifier avant le test

```text
Exports recensés + cohorte de robustesse fixée
  -> inférence commune audio -> fenêtres -> représentation -> scores
  -> reproduction des prédictions de validation
  -> sensibilité audio et traitement des entrées limites
  -> bootstrap vérifié sur exemples connus et validation
  -> contrôle des prérequis + manifeste final verrouillé
  -> une campagne test, avec reprise sans nouvel ajustement
  -> rapport final -> démo et reproduction portfolio
```

L’inférence commune du candidat existe désormais dans
`src/music_genre/inference/predict.py` ; le bloc 6 original est adapté dans
`adapters.py` ; les autres comparateurs linéaires sont vérifiés dans
`linear_adapters.py`. Les CNN sont vérifiés dans `cnn_adapter.py`. Elle réutilise les modules audio, extracteurs et exports existants.
L’inférence ne reçoit ni genre réel ni métadonnées d’artiste. L’évaluation
rapproche ensuite les résultats et les étiquettes par identifiant.

Pour FMA, conserver 16 kHz mono, fenêtres de dix secondes aux positions 0/10/20,
padding et moyenne des probabilités identiques au développement. Vérifier
l’ordre des huit classes et les empreintes des poids, configurations, scaler,
classificateur, sources et versions. Écart maximal autorisé avec les probabilités
de validation enregistrées : 10⁻⁶. Aucun ajustement de seuil après inspection du test.

Pour les fichiers de la future démo, distinguer un mode explicite début/milieu/fin
sur les longs fichiers du mode FMA fixe ; afficher les positions réellement
utilisées. Cette règle de démo ne doit jamais modifier l’évaluation FMA.

## Robustesse : un diagnostic borné, fixé avant calcul

Sélection de **64 morceaux de validation**, huit par genre et au plus un par
artiste, par le sélecteur déterministe existant `select_review`, graine 20261028.
La sélection utilise uniquement identifiant, genre et artiste, jamais les erreurs
ou scores des modèles. Le CSV conserve les empreintes des sources. Si un quota
ne peut être satisfait, documenter le manque avant calcul, sans remplacement
piloté par les résultats. Ce petit panel équilibré ne représente pas tout FMA.

Appliquer au candidat principal trois conditions, sur les mêmes morceaux :

| Condition | Définition | Ce que l’on mesure |
|---|---|---|
| Identité | Audio sans modification | Reproduction des probabilités enregistrées |
| Gain −6 dB | Multiplication par 10^(−6/20), sans renormalisation | Sensibilité au niveau sonore |
| Bruit à SNR 20 dB | Bruit blanc gaussien déterministe, RMS du bruit = RMS du signal / 10 | Sensibilité à une perturbation contrôlée |

Transformer le signal mono 16 kHz complet avant segmentation ; ne pas ajouter
de bruit au padding. Pour le bruit, normaliser le tirage à son RMS mesuré avant
application de l’échelle ; cas RMS nul traité explicitement. La graine par morceau
est l’entier représenté par les seize premiers chiffres hexadécimaux du SHA256
de `20261028:<track_id>:white_noise_snr_20_db` en UTF-8. Pas d’écrêtage ajouté ;
enregistrer le pic et la fraction des échantillons dépassant l’amplitude 1.

Rapporter macro-F1, accuracy, changements de classe, variation des probabilités,
résultats par genre, temps et erreurs de traitement, pour les trois conditions.
**Contrôles obligatoires :** 64 résultats complets, scores finis normalisés,
identité reproductible à 10⁻⁶. Pas d’élimination silencieuse d’un échec.

**Déclencheurs de revue fixés maintenant :** perte de macro-F1 d’au moins 0,05
ou changement de prédiction sur au moins 20 % des morceaux, face à l’identité.
Ce sont des seuils pratiques de diagnostic, pas une preuve statistique ni une
garantie produit. Un déclenchement impose d’expliquer la sensibilité avant
verrouillage ; il ne lance pas automatiquement une optimisation. Toute extension
du panel ou modification de modèle sera versionnée avant ouverture du test.

Les entrées limites relèvent de contrôles distincts : stéréo et rééchantillonnage,
extrait court non dupliqué, silence numérique, fichier vide/corrompu et limites
de ressources. Le module de diagnostic conserve actuellement le silence ; la
future fonction publique doit refuser un signal mono entièrement nul sans
inventer une classe. Ne pas créer de seuil de « musique » ou classe inconnue.
Prévoir 50 000 000 octets et 300 secondes maximum, et une borne de décodage
PCM temporaire de 128 Mio vérifiée avant allocation ; les fichiers dépassant
cette borne sont refusés explicitement. Ces protections publiques sont implémentées
et testées dans T1 ; le mode de segmentation de la démo reste à réaliser.

## Mesures finales et incertitude

Critère principal : macro-F1 **par morceau**, sur les huit classes fixées,
`zero_division=0`. Compléments : accuracy, précision/rappel/F1 par genre,
effectifs, matrice de confusion, erreurs, coût de calcul et latence. Une moyenne
par artiste pourra être présentée séparément, jamais substituée au score principal.

Différence principale annoncée : candidat moins ViT bloc 12 original.
Différences secondaires : fusion moins ViT original (hypothèse initiale),
candidat moins bloc 12 HPSS et candidat moins bloc 6 original. Pour le CNN,
publier chaque graine puis moyenne et écart-type des métriques ; ne pas choisir
la meilleure graine sur test ni transformer les trois réseaux en ensemble.

Bootstrap **groupé par artiste et apparié entre systèmes** : 2 000 répétitions,
PCG64 graine 20261028. À chaque répétition, tirer avec remise autant d’artistes
que d’artistes distincts dans la partition ; inclure tous leurs morceaux,
avec la multiplicité du tirage. Utiliser le même tirage pour tous les modèles.
Ne pas rééchantillonner les fenêtres ni les morceaux à l’intérieur de l’artiste.
Le score reste pondéré par morceaux ; le tirage des groupes ne produit pas
une moyenne égale par artiste.

Calculer les intervalles percentiles à 95 % (quantiles 0,025/0,975, interpolation
linéaire) des scores et des différences appariées. Pour la moyenne CNN, calculer
d’abord chaque score de graine sur le même tirage, puis leur moyenne. Garder
les huit classes même si une manque dans un tirage, et publier la fréquence
de ces tirages ; ne pas les supprimer ou les retirer jusqu’à réussite.
Pas de stratification par genre : un artiste peut porter plusieurs genres.

Ce choix produit des intervalles approximatifs, conditionnels aux modèles fixés,
pas une estimation de la variabilité de tout l’apprentissage. Les intervalles
secondaires sont ponctuels, sans correction simultanée : ne pas les transformer
en une collection de tests de significativité. Une graine, une méthode ou un
nombre de répétitions ne se choisit pas selon l’intervalle le plus favorable.

La nécessité de tenir compte des observations dépendantes est motivée par
[Saravanan, Berman et Sober (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290/).
Notre choix est un bootstrap de groupes à un seul niveau ; il ne reproduit pas
tout leur bootstrap hiérarchique. La [documentation SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
décrit l’appariement et les méthodes d’intervalle ; la fonction standard ne
déduit pas les groupes d’artistes et sa méthode par défaut est BCa, différente
du choix percentile explicite retenu ici. Ces références ne garantissent pas
une couverture exacte pour la macro-F1 sur FMA.

## Conditions de verrouillage et campagne test

Avant de changer l’état du protocole, exiger : inférence validée, rapport de
robustesse complet et déclencheurs examinés, bootstrap testé, comparateurs
vérifiés, partitions toujours séparées, versions et poids épinglés. Le
recensement préparatoire ne remplace pas ce contrôle. La configuration actuelle
porte volontairement `test_evaluation_allowed: false`.

Le nouveau manifeste de verrouillage contient les neuf exports exacts, sources
finales, paramètres, environnement, manifeste de données et preuves des contrôles.
Son pointeur vérifié est `reports/latest_final_evaluation_lock.json`. La configuration
préparatoire demeure inchangée ; seule la racine du nouveau manifeste porte
l’autorisation verrouillée. T7 doit enregistrer son code et sa répétition à blanc.
L’évaluateur refusera un artefact modifié, un protocole non verrouillé ou une
partition incohérente. Il n’appellera aucun `fit`, recherche de C ou sélection
d’époque. Toute panne sera tracée ; aucune exclusion fondée sur une mauvaise
prédiction. Reprendre une extraction identique est permis, mais changer le modèle
après avoir vu les résultats exige une nouvelle version et une évaluation
indépendante pour revendiquer une généralisation non guidée par ce test.

Livrables attendus : prédictions par morceau/fenêtre, métriques et intervalles,
comparaisons appariées, coût, provenance et limites, y compris résultats négatifs.
La revue subjective des 19 fiches reste non réalisée ; ce complément sera signalé
comme manquant s’il n’est pas disponible, sans fabriquer d’observation d’écoute.

Ordre d’implémentation et critères vérifiables : [plan](../tasks/plan.md) et
[tâches](../tasks/todo.md). Les scripts mentionnés dans les tâches comme futurs
ne sont pas encore des commandes disponibles.
