# Diagnostic de sensibilité du candidat

Cette étape exécute T2 du [protocole préparatoire](PROTOCOLE_EVALUATION_FINALE.md).
Elle mesure la sensibilité de bloc 6 HPSS à deux changements audio précis,
sans entraîner de modèle ni ouvrir le test. Les résultats numériques sont
publiés seulement après traitement complet dans `reports/ROBUSTNESS_PANEL.json`
et `reports/ROBUSTESSE_VALIDATION.md`.

## Résultats vérifiés le 28 octobre 2026

Suite réalisée : [T2a](../reports/DIAGNOSTIC_BRUIT.md) reproduit les résultats
et mesure les déplacements des représentations sans établir une cause unique.
[T2b est terminée](../reports/COMPARAISON_ROBUSTESSE_BLOC6.md) : perte F1 moindre pour l’original mais davantage de changements. Aucun gagnant global ; candidat HPSS conservé avec sensibilité acceptée comme limite. T3 à T5 sont depuis vérifiées ; prochaine action T6, manifeste de verrouillage.
La décision T2 ci-dessous conserve le contexte qui avait motivé T2a.

Les 64 morceaux ont été traités dans les trois conditions : 576 fenêtres au
total, en 7,46 minutes. Les neuf nouveaux tests et les contrôles réels passent.
La mesure révèle cependant une **sensibilité importante au bruit**.

| Condition | Macro-F1 | Morceaux corrects | Changements de classe |
|---|---:|---:|---:|
| Identité | 0,5479 | 35/64 | — |
| Gain −6 dB | 0,5050 | 32/64 | 6/64 (9,38 %) |
| Bruit SNR 20 dB | 0,3648 | 26/64 | 26/64 (40,63 %) |

Le gain réduit entraîne une correction et quatre régressions, plus un changement
entre deux classes incorrectes. Sa perte de macro-F1 (0,0429) reste sous 0,05,
mais n’est pas nulle. Le bruit produit une correction et dix régressions ; les
quinze autres changements restent des erreurs dans les deux conditions.
Sa perte de macro-F1 (0,1831) et ses changements de classe dépassent les deux
seuils prévus. **Les contrôles techniques réussis ne valident pas une robustesse
du classificateur au bruit.**

Sous bruit, les prédictions Electronic passent de 11 à 20, Folk de 10 à 2,
et Instrumental de 6 à 0. Sept des huit vrais Instrumental changent de classe,
ainsi que six Folk et cinq International. Aucun vrai Hip-Hop ne change de
classe, mais sa F1 recule à cause de faux positifs venant d’autres genres.
Ces observations localisent la fragilité sans en établir le mécanisme.

Les causes techniques déjà contrôlées : identité reproduite à 5,18 × 10⁻⁷,
SNR mesuré entre 19,999999995 et 20,000000006 dB, fenêtres identiques, WAV
FLOAT sans perte et originaux inchangés. Des amplitudes supérieures à 1 sont
déjà présentes dans l’identité ; aucun écrêtage supplémentaire n’a été ajouté.
Les comptes, matrices de confusion et macro-F1 ont été recalculés séparément
depuis les probabilités des 192 prédictions de morceau/condition.

**Décision :** T2 est terminée comme mesure, mais le verrouillage reste en
attente d’un diagnostic ciblé T2a. Sur les huit morceaux du pilote déjà fixés,
examiner les variations des canaux log-Mel original/H/P, les valeurs bornées
par le prétraitement, les distances entre embeddings et les marges de
classification. Ce sont des pistes à vérifier, pas des causes établies.
Budget du diagnostic : dix minutes de calcul ; aucun nouvel entraînement,
aucun réglage des seuils et aucune ouverture du test. Ses résultats détermineront
si une expérience corrective est justifiée ou si cette sensibilité doit être
conservée explicitement comme limite de la première version.

[Rapport chiffré](../reports/ROBUSTESSE_VALIDATION.md) ·
[Résultats détaillés](../reports/ROBUSTNESS_PANEL.json)

## Données et transformations

Le panel figé contient 64 morceaux de validation, huit par genre et au plus un
par artiste. Il a été choisi avant les perturbations, sans utiliser les erreurs
du modèle. Il ne représente pas la totalité des 780 morceaux de validation :
comparer les conditions entre elles sur ce panel, sans substituer leur score
au résultat de développement complet.

L’audio original est vérifié par empreinte puis décodé en mono 16 kHz. Trois
conditions utilisent le signal valide complet, avant segmentation :

- Identité : copie sans changement.
- Gain −6 dB : amplitude multipliée par 0,5011872336, sans renormalisation.
- Bruit blanc : tirage gaussien PCG64 propre à l’identifiant du morceau, puis
  mise à l’échelle pour un SNR mesuré de 20 dB sur le signal complet.

La règle de graine est celle du protocole ; les originaux ne sont pas modifiés.
Un signal nul rend ce SNR indéfini et provoque une erreur explicite. Aucun bruit
n’est ajouté au padding des fenêtres. Le SNR par fenêtre peut différer de
20 dB si l’énergie varie dans le morceau ; la cible porte sur le signal complet.
Le pic et la proportion d’échantillons dépassant l’amplitude 1 sont enregistrés,
sans écrêtage supplémentaire.

Les signaux transformés passent par des WAV FLOAT temporaires, dont la relecture
float32 est vérifiée bit à bit. La même méthode `CandidatePredictor.predict_file`
effectue ensuite découpage, HPSS, extraction et classification. Les temporaires
créés par cette exécution sont supprimés ; les résultats par morceau restent
dans son dossier daté sous `data/interim/`.

## Contrôles et budget

Les positions, durées valides et padding doivent rester identiques au contrôle
T1. L’identité est comparée aux probabilités historiques à 10⁻⁶. Les scores
doivent être finis, non négatifs et normalisés ; le SNR réel doit être à moins
de 10⁻⁴ dB de la cible. Les empreintes des entrées, sources et poids protègent
la cohérence de l’expérience. Les versions des bibliothèques sont enregistrées.

Le pilote prend le premier morceau sélectionné de chacun des huit genres,
selon l’ordre des labels. Il exécute les trois conditions ; ses résultats sont
conservés pour le panel complet. Un processus est utilisé. Le temps des huit
morceaux est multiplié par huit pour estimer le coût du panel ; une projection
supérieure à deux heures interrompt l’exécution avant les morceaux suivants.
Cette projection inclut le démarrage des premières opérations audio, mais
exclut le chargement initial du modèle ; elle n’est pas un benchmark matériel
contrôlé.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_robustness.py
```

Progression : `reports/ROBUSTNESS_PROGRESS.json`. Une erreur produit un état
`failed` et conserve les résultats partiels du dossier d’exécution. Il n’y a
pas de reprise automatique : une nouvelle invocation recommence le panel dans
un nouveau dossier. Un ancien rapport complet n’est une preuve actuelle que
si ses empreintes de sources, paramètres et données correspondent encore.

## Lecture des résultats

Le rapport présente scores, confusion et F1 par genre, corrections/régressions,
changements de classe et variations absolues des probabilités. Les déclencheurs
de revue restent ceux écrits avant calcul : perte de macro-F1 ≥ 0,05 ou
changements de classe sur ≥ 20 % des morceaux, face à l’identité.

Un déclenchement impose une analyse avant verrouillage. Il ne démontre pas un
bug et ne justifie pas automatiquement d’ajuster le modèle sur ces 64 morceaux.
L’absence de déclenchement ne prouve pas une robustesse générale. Avec huit
morceaux par genre, les variations par genre restent descriptives et peuvent
être sensibles à quelques exemples. Aucune cause auditive, significativité ou
performance sur de nouveaux artistes n’est déduite de ces seuls essais.
