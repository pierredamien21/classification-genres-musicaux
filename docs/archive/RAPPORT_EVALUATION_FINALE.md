# Rapport statistique des campagnes terminées

29 octobre 2026. Générateur implémenté ; son exécution de contrôle utilise les
sorties complètes de validation. Aucun calcul de métriques sur test partiel.

## Entrées et règles

`scripts/report_final_evaluation.py` accepte un dossier de campagne terminé avec
`campaign.json`, `completion.json`, `progress.json` et toutes les prédictions.
Il vérifie leur provenance et leurs empreintes, le manifeste verrouillé, les
versions, la partition entière, l’ordre des genres et les neuf identités de modèles.
Un dossier incomplet ou modifié est refusé. Aucun modèle n’est chargé pour prédire,
aucun audio n’est décodé et aucun paramètre n’est ajusté.

Les statistiques réutilisent les fonctions vérifiées de classification et de
bootstrap. L’assemblage est dans `src/music_genre/evaluation/final_reporting.py` :

- macro-F1 à huit labels, accuracy, précision/rappel/F1 par genre et confusion ;
- référence majoritaire déterminée par les effectifs **train** ;
- 2 000 tirages d’artistes avec multiplicité, PCG64 et graine 20261028 ;
- intervalles percentile ponctuels à 95 %, appariés entre systèmes ;
- comparaison principale et trois comparaisons secondaires du protocole ;
- moyenne des métriques CNN et écart-type entre graines, sans ensemble de probabilités ;
- nombre de tirages à classe absente, conservés avec labels fixes.

L’exécution de validation exige aussi une égalité exacte avec la preuve T3 déjà
vérifiée : scores, intervalles, différences et tableaux de rééchantillonnage.
Ce contrôle empêche une dérive de l’assemblage statistique avant le rapport test.

## Commandes

Commande de contrôle sur validation :

```powershell
.\.venv\Scripts\python.exe scripts/report_final_evaluation.py --campaign data/interim/final-validation-v1 --output reports/final-validation-report-v1
```

Commande à utiliser **après la fin de l’inférence test** :

```powershell
.\.venv\Scripts\python.exe scripts/report_final_evaluation.py --campaign data/interim/final-test-v1 --output reports/final-test-report-v1
```

Le dossier de rapport doit être nouveau : aucun rapport antérieur n’est écrasé.
Une nouvelle exécution avec un autre nom de sortie doit conserver les mêmes
preuves et paramètres. Le script vérifie les sources et les entrées avant
publication ; leurs empreintes sont incluses dans le rapport.

## Livrables et lecture

| Fichier | Contenu |
|---|---|
| `REPORT.md` | Tableau de scores/intervalles, différences appariées, candidat par genre, limites |
| `report.json` | Toutes les métriques, métadonnées, provenance, coûts et empreintes |
| `confusions.csv` | Lignes = genres réels, colonnes = genres prédits, par système |
| `per_class.csv` | Précision, rappel, F1 et effectif par genre et système |
| `bootstrap_draws.npz` | Multiplicités des artistes, scores des tirages et effectifs associés |

Les durées proviennent des appels d’inférence sauvegardés (somme, médiane, p95).
Elles ne couvrent pas tous les contrôles et chargements de la campagne. La durée
de la dernière invocation n’est pas une durée cumulée de plusieurs reprises.

Le statut statistique `complete` signifie que les fichiers du rapport sont
produits. `independent_report_check: pending` indique qu’ils attendent encore
leur contrôle indépendant final ; ne pas confondre ces deux étapes. La campagne
test reste en cours au moment de la préparation de cet outil.

## Vérifications et limites

Cinq tests spécifiques vérifient notamment le majoritaire train, la moyenne
des métriques CNN plutôt qu’un ensemble, une différence connue et le refus
de cohortes ou protocoles incompatibles. Suite complète : 257 tests réussis.
Les résultats de l’essai réel sont consignés dans
le résumé de partie.

Les intervalles sont conditionnels aux modèles fixés, sans rééchantillonnage
de l’apprentissage ni des graines, et sans correction pour tests simultanés.
Sur validation, la sélection des modèles reste une limite. Les résultats test
ne devront guider ni nouveau choix de paramètres ni exclusion de morceau.
Les limites audio, les huit genres fermés et l’absence de revue subjective sont
conservés dans le rapport. Les figures du notebook pourront exploiter ces exports.

## Contrôle test terminé le 29 octobre

`reports/final-test-report-v1/` a été généré puis vérifié. Commande :

```powershell
.\.venv\Scripts\python.exe -u scripts/verify_final_test.py --report reports/final-test-report-v1/report.json
```

La preuve `reports/FINAL_TEST_REPORT_CHECK.json` porte `status: passed` et
l’empreinte du rapport brut. Celui-ci conserve son état historique `pending`
à la génération : ne pas le réécrire après contrôle. 7 002 prédictions et
2 000 tirages vérifiés, 44 000 métriques identiques. Le pointeur
`reports/latest_final_test_report.json` relie les artefacts vérifiés.
[Interprétation finale](../../reports/EVALUATION_FINALE.md).
