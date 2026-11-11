# Robustesse comparée : bloc 6 original et bloc 6 HPSS

Mis à jour : 28 octobre 2026. T2b terminée. Comparaison descriptive sur les mêmes 64 morceaux de validation (huit par genre, 64 artistes), avec les trois conditions et graines fixées en T2. Les scores de ce panel ne sont pas ceux des 780 morceaux de validation. Jeu de test non ouvert.

## Résultat et décision

Les critères annoncés ne convergent pas : le bloc 6 original perd moins de macro-F1 sous bruit, mais change davantage de classes. Aucun gagnant global en robustesse n’est établi. Les deux systèmes franchissent les seuils de revue sous bruit. Cette comparaison ne soutient pas une fragilité exclusivement liée à HPSS ; les têtes de classification ayant été ajustées séparément, elle n’isole pas un effet causal des canaux.

Pour la version actuelle, conserver le candidat HPSS choisi sur la validation complète et accepter explicitement la sensibilité au bruit comme limite mesurée. Aucun débruitage, nouvelle normalisation, réentraînement ou changement des seuils. Poursuivre T3 (incertitude par artiste), puis les comparateurs restants, avant verrouillage et test. Une expérience corrective ultérieure devra annoncer ses paramètres et critères avant calcul ; elle n’est pas lancée ici.

## Scores et compromis

| Système | Condition | Macro-F1 | Accuracy | Perte F1 vs propre | Classes changées | Corrections / régressions |
|---|---|---:|---:|---:|---:|---:|
| Bloc 6 original | Identité | 0.4693 | 0.4688 | +0.0000 | 0/64 | 0 / 0 |
| Bloc 6 original | Gain −6 dB | 0.4731 | 0.4688 | -0.0037 | 4/64 | 1 / 1 |
| Bloc 6 original | Bruit SNR 20 dB | 0.3681 | 0.3750 | +0.1012 | 28/64 | 3 / 9 |
| Bloc 6 HPSS | Identité | 0.5479 | 0.5469 | +0.0000 | 0/64 | 0 / 0 |
| Bloc 6 HPSS | Gain −6 dB | 0.5050 | 0.5000 | +0.0429 | 6/64 | 1 / 4 |
| Bloc 6 HPSS | Bruit SNR 20 dB | 0.3648 | 0.4062 | +0.1831 | 26/64 | 1 / 10 |

La baisse sous bruit est de 0,1012 pour l’original contre 0,1831 pour HPSS ; l’original part toutefois de plus bas sur ce panel (0,4693 contre 0,5479). Les macro-F1 bruitées sont proches (0,3681 contre 0,3648), tandis que l’accuracy bruitée favorise HPSS (24/64 contre 26/64 corrects). Les changements sont 28/64 (43,75 %) contre 26/64 (40,625 %). Une différence de deux morceaux ou de quelques millièmes ne prouve pas une amélioration généralisable.

Le gain −6 dB provoque quatre changements et une accuracy inchangée pour l’original, contre six changements et trois bonnes réponses perdues nettes pour HPSS. Le score absolu HPSS reste supérieur sous gain. Une perte de F1 négative indique ici une légère hausse, pas une erreur de signe.

## Résultats par genre

Chaque genre comporte seulement huit morceaux. Les changements sont comptés relativement à la prédiction propre de chaque modèle ; ils incluent les passages entre deux classes incorrectes.

| Genre | F1 original propre | F1 original gain | F1 original bruit | F1 HPSS propre | F1 HPSS gain | F1 HPSS bruit | Changements bruit original / HPSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Electronic | 0.4762 | 0.4000 | 0.3243 | 0.6316 | 0.4706 | 0.5000 | 1 / 1 |
| Experimental | 0.4000 | 0.4000 | 0.2500 | 0.5333 | 0.4706 | 0.3333 | 5 / 3 |
| Folk | 0.4211 | 0.5556 | 0.2222 | 0.5556 | 0.5263 | 0.4000 | 5 / 6 |
| Hip-Hop | 0.8000 | 0.8000 | 0.7143 | 0.7059 | 0.7500 | 0.5714 | 1 / 0 |
| Instrumental | 0.2857 | 0.2353 | 0.0000 | 0.2857 | 0.3750 | 0.0000 | 5 / 7 |
| International | 0.4706 | 0.4706 | 0.5455 | 0.6667 | 0.6667 | 0.3636 | 3 / 5 |
| Pop | 0.2857 | 0.3077 | 0.2222 | 0.2353 | 0.2353 | 0.1250 | 4 / 2 |
| Rock | 0.6154 | 0.6154 | 0.6667 | 0.7692 | 0.5455 | 0.6250 | 4 / 2 |

## Reproduction et contrôles

L’export original déjà entraîné est chargé par `OriginalBlock6Predictor`. Il partage le contrat audio du candidat, réutilise `MusicalViT` et applique le préfixe de six blocs avec la même LayerNorm terminale. Les 64 prédictions propres reproduisent les références sauvegardées avec un écart maximal 4.4662986e-07, inférieur à 10⁻⁶. Les 576 nouvelles fenêtres conservent les positions et durées valides de T2.

Les résultats HPSS sont réutilisés depuis leur rapport vérifié et leurs métriques sont recalculées à partir des prédictions. Les empreintes des sources, poids, exports, manifeste et panel sont contrôlées. Le même code de transformation et les mêmes données brutes produisent les signaux ; WAV FLOAT sans perte et absence d’écrêtage ajouté. Les 192 empreintes PCM originales sont enregistrées pour reproduction.

Pilote : 6.09 secondes pour huit morceaux, projection 48.69 secondes. Exécution : 57.73 secondes, sous le budget de trente minutes. Coûts indicatifs sur ce poste ; les durées historiques HPSS ne constituent pas un benchmark comparatif contrôlé.

155 tests passent. Vérification indépendante des matrices de confusion, macro-F1, accuracy, corrections/régressions, moyennes des fenêtres, empreintes et SNR : [preuve](BLOCK6_ROBUSTNESS_CHECK.json). Les tests locaux de l’adaptateur demandent les exports présents ; aucun téléchargement automatique.

Commandes :

```powershell
.\.venv\Scripts\python.exe scripts/compare_block6_robustness.py
.\.venv\Scripts\python.exe scripts/check_block6_robustness_comparison.py
```

## Limites

Une seule perturbation de bruit blanc, un SNR et une graine par morceau ; aucun bruit réel, codec ou autre corpus testé. Panel limité, pas d’intervalle ni de significativité revendiqués. Résultats déjà consultés sur validation : pas de généralisation indépendante. La revue subjective reste à compléter. La robustesse générale du modèle n’est pas démontrée.

[Résultats et provenance](BLOCK6_ROBUSTNESS_COMPARISON.json) · [Diagnostic préalable](DIAGNOSTIC_BRUIT.md) · Plan
