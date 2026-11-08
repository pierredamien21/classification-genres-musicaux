# Évaluation finale V1 — résultats vérifiés

29 octobre 2026. T7 terminée. Le candidat choisi sur validation, **ViT bloc 6 +
HPSS**, obtient **0,5035 de macro-F1 et 51,16 % d’accuracy** sur 778 morceaux
test appartenant à 202 artistes (398 classifications correctes).

## Protocole et contrôle

Neuf exports fixés avant le test, sans réentraînement ni sélection supplémentaire.
Partitions conservées : 6 299 morceaux train, 780 validation, 778 test.
Trois fenêtres par morceau, agrégation des probabilités puis décision par morceau.
Le majoritaire est déterminé sur train : Experimental.

Bootstrap apparié par artiste : 2 000 tirages PCG64, graine 20261028,
artistes entiers avec multiplicité, intervalles percentile à 95 %. Les intervalles
sont ponctuels, conditionnels aux modèles fixés, sans correction des comparaisons
multiples ni rééchantillonnage de l’entraînement. Aucun tirage ne manque de classe.

Le contrôle distinct a vérifié 7 002 prédictions, 21 006 vecteurs de fenêtres,
les empreintes et les confusions. Il a ensuite reconstruit les tirages en développant
explicitement les morceaux des artistes : 44 000 métriques concordent exactement.
Suite complète : 261 tests réussis ; Ruff réussi sur le nouveau vérificateur et ses tests.

## Résultats test

| Modèle | Macro-F1 | IC 95 % macro-F1 | Accuracy |
|---|---:|---:|---:|
| Majoritaire train | 0,0277 | [0,0157 ; 0,0437] | 12,47 % |
| Descripteurs musicaux | 0,3892 | [0,3011 ; 0,4260] | 39,59 % |
| ViT bloc 12 original | 0,4641 | [0,3761 ; 0,5073] | 47,56 % |
| Fusion ViT + descripteurs | 0,4791 | [0,3892 ; 0,5142] | 48,71 % |
| ViT bloc 6 original | 0,4883 | [0,3991 ; 0,5294] | 49,49 % |
| ViT bloc 12 HPSS | 0,4694 | [0,3844 ; 0,5108] | 47,94 % |
| **ViT bloc 6 HPSS — candidat** | **0,5035** | **[0,4133 ; 0,5449]** | **51,16 %** |
| CNN graine 42 | 0,4539 | [0,3643 ; 0,5003] | 46,02 % |
| CNN graine 43 | 0,4424 | [0,3536 ; 0,4976] | 46,40 % |
| CNN graine 44 | 0,3995 | [0,3069 ; 0,4502] | 41,65 % |
| Moyenne des métriques CNN | 0,4320 | [0,3451 ; 0,4796] | 44,69 % |

Écart-type échantillonnal entre les trois graines CNN : 0,0287 en macro-F1,
0,0264 en accuracy. Cette moyenne ne constitue pas un ensemble de probabilités.

## Comparaisons annoncées

Différences de macro-F1 en points sur une échelle de 0 à 100 :

| Comparaison | Différence | IC apparié 95 % |
|---|---:|---:|
| Candidat − ViT original (principale) | +3,94 | [+0,54 ; +7,15] |
| Fusion − ViT original | +1,50 | [−2,77 ; +5,08] |
| Candidat − ViT bloc 12 HPSS | +3,41 | [−0,06 ; +6,58] |
| Candidat − ViT bloc 6 original | +1,52 | [−1,12 ; +4,06] |

La comparaison principale soutient un gain de macro-F1 face au ViT original
dans ce protocole. Les trois intervalles secondaires incluent zéro : ils ne
permettent pas d’affirmer un gain établi. Même pour la comparaison principale,
l’intervalle de différence d’accuracy inclut zéro (−0,13 à +7,11 points).
Les conclusions viennent des différences appariées, pas du chevauchement des
intervalles individuels. Le petit gain ponctuel de fusion sur test contraste
avec sa baisse sur validation ; il n’établit pas sa supériorité et ne motive
aucune nouvelle sélection.

## Genres et limites

| Genre | Effectif | Corrects | F1 candidat |
|---|---:|---:|---:|
| Electronic | 95 | 67 | 0,6351 |
| Experimental | 97 | 30 | 0,3550 |
| Folk | 100 | 39 | 0,3714 |
| Hip-Hop | 98 | 76 | 0,7525 |
| Instrumental | 96 | 36 | 0,4186 |
| International | 100 | 55 | 0,5288 |
| Pop | 92 | 29 | 0,3258 |
| Rock | 100 | 66 | 0,6408 |

Parmi les confusions du candidat : 26 Instrumental vers Folk, 20 Experimental
vers Instrumental, 20 International vers Folk et 19 Folk vers Pop. Ce sont des
observations quantitatives, sans explication acoustique causale ni écoute subjective
réalisée. Les matrices complètes sont dans les exports.

La macro-F1 du candidat passe de 0,5975 sur validation à 0,5035 sur test
(−9,40 points). Le score de sélection surestimait donc ici le score test observé ;
la cause précise de cette différence n’est pas établie. FMA-small, huit classes
et artistes retenus ne représentent pas toute la musique. Le gospel, le multigenre
et la détection des genres inconnus ne sont pas couverts. Scores non calibrés.
La sensibilité au bruit constatée avant verrouillage reste une limite.

## Coût et suite

Campagne d’inférence : 51 min 50 s sur le CPU local, sans CUDA. Pour le candidat,
somme des appels enregistrés 848,84 s, médiane 1,092 s par morceau, p95 1,115 s.
Ces mesures n’incluent pas tous les chargements et contrôles et ne prédisent pas
la latence de la future interface. Aucun nouvel entraînement lancé ici.

Restent les notebooks pédagogiques, la démo locale, la reproduction dans un
environnement propre et la présentation portfolio. La session autonome demandée
(environnement distinct, plafond 20/30 époques et arrêt anticipé) sera préparée
avant la conclusion générale ; l’utilisateur la lancera puis signalera sa fin.
Toute extension après consultation du test V1 reste du développement complémentaire.

## Preuves

- [Rapport brut et paramètres](final-test-report-v1/report.json), [tableaux](final-test-report-v1/REPORT.md).
- [Confusions](final-test-report-v1/confusions.csv), [métriques par genre](final-test-report-v1/per_class.csv).
- [Recomptage indépendant](FINAL_TEST_RECOUNT.json), [contrôle statistique](FINAL_TEST_REPORT_CHECK.json).
- [Manifeste verrouillé](final-evaluation-lock-20261029T085227817846Z.json), [suivi de campagne](FINAL_TEST_JOB.json).
- [Fiche du modèle](../docs/FICHE_MODELE.md), [résumé de reprise](../docs/resumes/37_EVALUATION_FINALE.md).

Le rapport brut conserve son état historique `independent_report_check: pending`
à sa création. Le contrôle externe ultérieur porte l’état `passed` et l’empreinte
de ce rapport. Le brut n’a pas été réécrit après vérification.
