# Combinaison bloc 6 + HPSS

Protocole fixé le 27 octobre 2026, avant extraction complète et scores du candidat.

## Résultat complet vérifié le 28 octobre 2026

L’extraction, l’entraînement et la comparaison sont terminés : 7 079 morceaux,
21 237 fenêtres, C = 0,1 retenu. Macro-F1 validation **0,5975**, accuracy
**0,6128** : +0,0131 en macro-F1 face à bloc 12 HPSS et +0,0195 face à bloc 6
original. Face au premier : 56 corrections et 44 régressions ; certains genres
régressent. Le test n’a pas été évalué. Voir le [rapport complet](../reports/COMBINAISON_BLOC6_HPSS.md).

Extraction : 33,35 minutes à quatre processus dans l’invocation finale, 64
morceaux repris du pilote ; entraînement : 4,71 s. L’export rechargé reproduit
les prédictions sauvegardées à 2,22 × 10⁻¹⁶ près. Les rapports de progression
indiquent `complete`. Les passages ci-dessous décrivent le protocole et les
mesures préalables ; ils ne remplacent pas le résultat final.

## Question et comparateurs

Les canaux original/harmonique/percussif améliorent le bloc 12 et le bloc 6
améliore la référence à canaux répétés. Leur combinaison n’a pas été évaluée.
La question est de savoir si elle apporte un gain par rapport à chacune des
deux variantes, sans présumer que les améliorations s’additionnent.

- Bloc 6 + HPSS contre bloc 12 + HPSS : seule la profondeur change.
- Bloc 6 + HPSS contre bloc 6 original : seuls les canaux changent.
- Bloc 12 original : repère historique, deux facteurs changent avec le candidat.

Les mêmes 6 299 morceaux de train et 780 de validation sont prévus ; le test
reste réservé. Trois fenêtres de dix secondes, mono 16 kHz, HPSS avant Mel,
canaux original/harmonique/percussif, plage fixe −80 à 20 dB et redimensionnement
224 × 224 sont conservés. Les six premiers blocs du checkpoint local sont gelés,
suivis de sa LayerNorm finale inchangée et de la moyenne des tokens de patches,
sans les préfixes : vecteur de 192 valeurs.

La standardisation sera apprise sur train uniquement. La grille logistique
C = 0,1 / 1 / 10 et la sélection par macro-F1 validation, égalité au plus petit C,
restent inchangées. Les probabilités des trois fenêtres sont moyennées. Rapporter
les différences avec les deux comparateurs, les F1 par genre et le coût ; ne pas
retenir uniquement les genres améliorés. Un petit gain sur validation ne prouve
pas une amélioration sur de nouveaux artistes.

## Contrôles avant extraction

Vérifier que les spectrogrammes sont identiques à ceux du bloc 12 HPSS et que
la sortie est celle du sixième bloc de ce même réseau, suivie de sa LayerNorm.
Avec trois copies du canal original, le candidat doit reproduire le cache bloc 6
original à 1e-6 près. Le bloc 12 avec les vrais canaux doit reproduire le cache
HPSS existant à 1e-6 près. Le canal original doit reproduire le log-Mel natif à
1e-5 dB près. Sources audio, paramètres, versions et poids sont contrôlés.

## Benchmark préalable

64 fenêtres centrales de train : huit morceaux par genre, graine 42, après une
fenêtre de chauffe exclue des mesures. Un processus, un fil de calcul. Séparer
décodage, séparation HPSS/log-Mel et encodeur bloc 6 ; exclure les comparaisons
de contrôle du chronométrage du candidat. Mesurer aussi le bloc 12 HPSS sur les
mêmes canaux pour ne pas comparer seulement des mesures faites à des dates
différentes.

Estimation séquentielle : coût moyen de décodage × 7 079 morceaux + coût moyen
HPSS/ViT × 21 237 fenêtres. Appliquer un facteur de prudence de deux, puis vérifier
un budget de douze heures. Ce calcul n’est pas une durée garantie : écritures,
initialisation, interruptions et charge du poste ne sont pas entièrement couverts.
Le parallélisme devra être mesuré séparément, sans diviser arbitrairement la durée.

## Résultat du benchmark

Benchmark terminé : 64 fenêtres de train, toutes de dimension 192 et finies.
Écarts maximaux avec le log-Mel original, le cache bloc 6 à canaux répétés et le
cache bloc 12 HPSS : **0**. Le test avec le réseau complet confirme la sortie au
sixième bloc et la conservation exacte des spectrogrammes. Les **113 tests**
passent ; Ruff passe. Aucun code de référence ni cache antérieur n’a été modifié.

| Opération | Moyenne | Médiane |
|---|---:|---:|
| Décodage, rééchantillonnage et découpage central | 0,07363 s | 0,07398 s |
| HPSS et log-Mel | 0,31243 s | 0,31212 s |
| ViT bloc 6 | 0,03321 s | 0,03298 s |
| ViT bloc 12 sur les mêmes canaux, contrôle séparé | 0,06045 s | 0,06035 s |
| Candidat : décodage + HPSS + bloc 6 | 0,41927 s | 0,41900 s |

Le réseau au bloc 6 prend environ 45 % de temps en moins que le bloc 12 dans
cette mesure appariée. Toutefois, HPSS/log-Mel représente environ 90 % du coût
HPSS + bloc 6 : le coût total décodage inclus ne baisse que d’environ 6 % face
au bloc 12 mesuré sur ces mêmes canaux. Ces nombres ne mesurent pas la précision.

Projection séquentielle pour 7 079 morceaux et 21 237 fenêtres : **7 861 s**,
soit **2 h 11**. Avec le facteur de prudence de deux : **15 723 s**, soit
**4 h 22**, inférieur au budget fixé de douze heures. L’extraction complète
n’est pas réalisée ; le coût réel reste à mesurer, notamment les écritures.

Une première mesure chevauchait l’exécution des tests et donnait 14 691 s de
projection brute. Elle est conservée dans
`reports/VIT_BLOCK6_HPSS_BENCHMARK_FIRST_RUN.json`. La mesure retenue est une
répétition effectuée après la fin des tests, avec exactement le même code, les
mêmes données et les mêmes seuils. Cette variation illustre la sensibilité au
poste ; elle ne permet pas d’attribuer tout l’écart à une seule cause.

## Preuves et suite

- [Rapport détaillé, mesures et empreintes](../reports/VIT_BLOCK6_HPSS_BENCHMARK.json).
- [Pointeur vérifié](../reports/latest_vit_block6_hpss_benchmark.json).
- [Script du benchmark](../scripts/benchmark_vit_block6_hpss.py).
- [Tests de la combinaison](../tests/test_hpss_block6.py).

Reproduire depuis la racine, sans lancer d’autres calculs simultanément :

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_vit_block6_hpss.py
```

Extraction complète reprenable implémentée le 27 octobre 2026 et relancée
le 28 dans une session suivie, après un démarrage détaché non persistant.
Le lanceur enchaîne ensuite la grille logistique et la comparaison aux deux
références, uniquement après validation du cache. Le cache complet, le classificateur et les scores sont désormais disponibles ;
le benchmark seul ne préjugeait pas du gain.

Configurations : `configs/vit_block6_hpss.json`,
`configs/baseline_vit_block6_hpss.json`, `configs/block6_hpss_experiment.json`.

## Exécution complète et reprise

```powershell
.\.venv\Scripts\python.exe scripts/run_vit_block6_hpss.py --workers 4
```

Le traitement actif est exécuté dans une session de calcul suivie, avec sorties
persistantes référencées par `reports/latest_block6_hpss_launch.json`. Lire
`reports/VIT_BLOCK6_HPSS_PIPELINE.json` et `reports/VIT_BLOCK6_HPSS_PROGRESS.json`
avant une relance. Un verrou Windows empêche deux lanceurs concurrents ; il
est libéré automatiquement à la fin du processus. Ne pas modifier les sources
pendant l’extraction : leurs empreintes sont vérifiées avant publication.

Pilote : 64 morceaux de train, trois fenêtres chacun, 37,7 s à quatre processus,
initialisation comprise. Reprise du même lot : 64/64 morceaux relus sans recalcul,
9,5 s. Les fichiers terminés sont protégés par une empreinte et les identités
source/configuration ; un fragment sans marqueur peut être recalculé, une
empreinte incohérente provoque un arrêt. Le cache final reprend l’index exact des
références. Les 120 tests passent ; Ruff passe.

L’option `vit_block6_hpss` a été ajoutée au parseur du script d’entraînement
commun ; aucune logique d’ajustement, de sélection ou d’export n’a changé. Les
rapports historiques conservent l’empreinte de l’ancienne version du script.
L’extraction et les modèles historiques ne sont pas réécrits.

À l’achèvement, les sorties prévues sont `reports/latest_vit_block6_hpss_run.json`
et `reports/COMBINAISON_BLOC6_HPSS.md`. Leur absence pendant l’extraction est
normale. Un état `failed` ou `interrupted` demande une inspection des journaux ;
il ne doit pas être présenté comme une expérience terminée. La suite de
comparaison vérifie les paramètres, l’alignement, les métriques et les éventuels
avertissements d’ajustement, et rapporte également les régressions.
