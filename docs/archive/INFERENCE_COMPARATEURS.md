# Inférence des comparateurs linéaires

État au 28 octobre 2026 : T4 vérifiée, 196 tests réussis.
[Résultats et preuves](../../reports/INFERENCE_COMPARATEURS.md).

Les quatre adaptateurs ajoutés en T4 utilisent les exports déjà entraînés :
descripteurs (125 dimensions), ViT bloc 12 original (192), fusion (317) et
ViT bloc 12 HPSS (192). Le bloc 6 original et le candidat bloc 6 HPSS disposent
déjà de leurs interfaces. Aucun ajustement de scaler ou de classificateur.

## Utilisation

```python
from pathlib import Path
from music_genre.inference.linear_adapters import LinearComparatorPredictor

root = Path("E:/projet/vision transformer")
predictor = LinearComparatorPredictor(root, "fusion")
# Remplacer par un fichier WAV/MP3 existant.
result = predictor.predict_file(root / "data/interim/mon_extrait.wav", mode="fma")
print(result["top3"])
```

Systèmes acceptés : `descriptor`, `vit`, `fusion`, `vit_hpss`.
Pour `vit_block6`, utiliser `OriginalBlock6Predictor` dans `inference/adapters.py` ;
pour `vit_block6_hpss`, utiliser `CandidatePredictor` dans `inference/predict.py`.

Le contrat de fichier, les limites de décodage, le refus du silence, la segmentation
FMA et le format de résultat sont ceux de [l’inférence du candidat](../INFERENCE_CANDIDAT.md).
Les extracteurs utilisent les configurations des rapports de caractéristiques
épinglés, et non des configurations courantes susceptibles d’avoir changé.
L’API demande ces rapports, les exports et les poids ViT éventuels ; elle ne
demande ni matrice de caractéristiques ni étiquette du morceau fourni.

## Fusion et ordre des caractéristiques

`AlignedFusion` transmet le même objet `AudioWindow` au ViT et aux descripteurs.
Les deux branches ne construisent donc pas séparément leurs listes de fenêtres.
La configuration doit annoncer l’ordre ViT puis descripteurs, toutes les colonnes
de descripteurs dans leur ordre initial, et la même fréquence, durée et positions.
Les noms et familles de caractéristiques sont comparés à ceux de l’export avant
la première prédiction. L’unique scaler chargé reste celui de la fusion entraînée.

Le concaténateur historique de caches exige une partition train/validation.
L’inférence ne fabrique pas de partition ou d’étiquette pour le réutiliser : elle
réutilise les extracteurs, aligne par la fenêtre commune et vérifie les colonnes.
Les noms, l’ordre et les dimensions sont contrôlés avant l’appel au classificateur.

## Reproductibilité numérique

L’extraction historique limitait les bibliothèques numériques à un thread.
`LinearComparatorPredictor.predict_file` applique cette même limite dans un
contexte temporaire, puis restaure le réglage de l’appelant. Le test de régression
inclut un appelant avec huit threads BLAS. Les limites de threads concernent
le processus : les appels concurrents dans un même processus ne sont pas testés.
Le vérificateur T4 applique également ce contexte aux deux interfaces historiques,
sans réécrire leurs sources ni leurs preuves antérieures.

Les anciennes durées de padding ont été calculées par soustraction, tandis que
l’API divise un nombre entier d’échantillons par la fréquence. Le vérificateur
compare la géométrie en échantillons entiers ; seul l’arrondi flottant inférieur
à 10⁻⁷ échantillon est admis. Un décalage d’un échantillon reste détecté.

## Contrôle de validation

Commande : `.\.venv\Scripts\python.exe scripts/check_comparator_inference.py`.

Le script contrôle les six systèmes linéaires, le panel fixé de 64 morceaux et
les probabilités par fenêtre et morceau (tolérance 10⁻⁶). Il recalcule aussi les
probabilités et métriques des 780 morceaux de validation depuis les matrices
existantes et les modèles exportés. Les index doivent correspondre exactement
entre systèmes et au manifeste ; les empreintes des sources et artefacts sont
vérifiées avant et après. Le jeu de test n’est pas chargé.

Le rapport d’exécution fait foi pour l’état réel :
`reports/COMPARATOR_INFERENCE_CHECK.json`, avec progression séparée dans
`reports/COMPARATOR_INFERENCE_PROGRESS.json`. La vérification CNN, la CLI et
la démo restent des étapes distinctes.
