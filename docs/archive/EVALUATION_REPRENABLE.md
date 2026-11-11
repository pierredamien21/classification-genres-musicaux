# Évaluation reprenable — T7

L’évaluateur applique les neuf exports du protocole verrouillé. Il ne contient
aucun entraînement ni sélection de paramètres. La première campagne porte sur
les 780 morceaux de validation ; le test exige sa réussite préalable.

## Commandes

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe -u scripts/evaluate_final.py --split validation --output data/interim/final-validation-v1
```

Relancer cette même commande reprend les prédictions manquantes. Avant une
relance, vérifier que le processus précédent est arrêté. Le verrou du système
d’exploitation refuse deux écrivains simultanés et se libère en cas d’arrêt ;
la présence du fichier `.writer.lock` ne signifie pas qu’un processus tourne.

Après réussite et contrôle de la validation, la commande test est :

```powershell
.\.venv\Scripts\python.exe -u scripts/evaluate_final.py --split test --output data/interim/final-test-v1 --validation data/interim/final-validation-v1
```

Cette commande a été lancée le 29 octobre après le contrôle indépendant de
validation : [suivi de la campagne test](../resumes/35_CAMPAGNE_TEST.md). Elle exige le
manifeste verrouillé, les mêmes sources d’évaluation et un dossier de validation
complet dont les empreintes sont encore valides.

## Traçabilité et reprise

Chaque campagne enregistre dans `campaign.json` l’empreinte du verrouillage,
les sources de l’évaluateur, les identités des neuf modèles, l’ordre des genres,
la partition entière, les fichiers audio et les réglages numériques. Une reprise
avec des paramètres ou sources différents échoue avant tout nouveau décodage.
Le contrôle initial vérifie les 7 985 artefacts du verrouillage, puis chaque audio
est de nouveau haché avant son utilisation. Les bibliothèques doivent conserver
leurs versions verrouillées.

Les résultats se trouvent dans `predictions/<système>/<track_id>.json`. Chaque
enregistrement contient les probabilités des trois fenêtres et leur moyenne,
la provenance de la campagne et une empreinte du contenu. L’écriture est atomique :
un fichier temporaire interrompu ne devient pas une prédiction valide. Un résultat
finalisé n’est pas écrasé. La reprise contrôle son contenu et réutilise le résultat
seulement s’il est cohérent. Les fichiers JSON supplémentaires sont refusés lors
du contrôle de complétude.

Sur validation, chaque système est comparé aux probabilités historiques par
identifiant. Tous les morceaux sont exigés ; tolérance absolue 10⁻⁶, mêmes classes
prédites. Un échec arrête la campagne et conserve les fichiers pour diagnostic.
Il ne provoque aucune exclusion de morceau ni modification du modèle.

`completion.json` n’est produit qu’après complétude, nouveau contrôle des sources,
des versions et de tous les artefacts. Il contient les empreintes des prédictions.
La commande test contrôle ce document et tous les résultats de validation avant
l’accès au test. Un marqueur `test_access_started.json` trace cet accès avant le
premier décodage, même si ce dernier échoue.

## Suivi sans attente active

- `progress.json` : phase, état, PID, compteurs et estimation du travail restant.
- `events.jsonl` : progression conservée toutes les dix nouvelles prédictions.
- `invocations.jsonl` : commandes, dates et identités de campagne lors des reprises.
- `errors.log` : erreur complète ; les résultats déjà valides restent disponibles.
- `reports/FINAL_VALIDATION_JOB.json` : session, journaux et informations du lancement.

La première estimation est de 1 h 35–3 h 10 pour l’inférence de validation,
hors empreintes et comparaison finale. Elle vient de huit morceaux train et
peut varier avec la charge et la chauffe de la machine. Le compteur porte sur
7 020 couples modèle/morceau, soit 21 060 vecteurs de fenêtres.

Une surveillance horaire avait été configurée dans cette conversation : notification
uniquement à la fin, en cas d’échec ou si une action est requise. Elle a été supprimée après la réussite du 29 octobre à 17 h 02 (Paris). Elle n’a pas lancé
le test. Les fichiers de progression permettent aussi une lecture sur demande.
Une absence de mise à jour doit être examinée avec le processus et les journaux :
elle ne suffit pas à conclure à une panne pendant le contrôle d’intégrité.

## État des vérifications et limites

16 tests spécifiques au stockage et cinq au recomptage indépendant, soit
252 tests au total, passent ; Ruff est vérifié. Les tests
exercent les reprises, les provenances incompatibles, les prédictions incorrectes,
les écritures interrompues, le verrou concurrent et les exigences de validation.
La répétition complète s’est terminée avec succès : 7 020 prédictions, neuf
systèmes sans désaccord de classe avec les références. Code de sortie 0 ; le
recomptage indépendant des sorties a réussi. La campagne test est lancée ;
son rapport statistique reste à produire.

Cette étape sauvegarde les prédictions et contrôle leur fidélité. Le générateur
du [rapport statistique](RAPPORT_EVALUATION_FINALE.md) est vérifié sur validation.
Son application au test et sa vérification indépendante restent à réaliser : majoritaire de train, métriques par genre, confusions, bootstrap
par artiste, comparaisons annoncées, moyenne des métriques CNN et contrôle
indépendant. Les empreintes détectent les modifications accidentelles ; elles
ne remplacent pas une signature ni un environnement stable pendant la campagne.
