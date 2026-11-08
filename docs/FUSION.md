# Fusion de caractéristiques et ablations

## Hypothèse et résultat

Les descripteurs résument explicitement timbre, rythme et classes de hauteur ; le
ViT fournit une représentation apprise sur des images puis appliquée aux log-Mel.
L’hypothèse est qu’ajouter les premiers aux seconds améliore la classification.
La présente expérience teste une **concaténation suivie d’un classifieur linéaire**.
Elle ne teste ni une fusion de probabilités ni un mécanisme d’attention multimodale.

Sur les 780 morceaux de validation, la fusion complète obtient une macro-F1 de
**0,5219**, contre **0,5447** pour le ViT seul. Les variantes sans rythme (**0,5387**)
et sans harmonie (**0,5282**) ne le dépassent pas non plus. L’hypothèse de gain
n’est pas soutenue dans ce protocole. Les résultats détaillés, y compris les
essais moins performants, sont conservés dans
[le rapport comparatif](../reports/COMPARAISON_FUSION.md).

## Construction des vecteurs

Les caches existants évitent de recalculer les spectrogrammes et les descripteurs.
Pour chaque fenêtre, le vecteur commence par les 192 dimensions ViT et se poursuit
par les dimensions retenues des descripteurs, dans leur ordre d’origine.

| Variante | ViT | Timbre/énergie | Harmonie (chroma) | Rythme (ACF) | Total |
|---|---:|---:|---:|---:|---:|
| Fusion complète | 192 | 60 | 24 | 41 | 317 |
| Sans rythme | 192 | 60 | 24 | 0 | 276 |
| Sans harmonie | 192 | 60 | 0 | 41 | 293 |

L’assemblage ne calcule aucune statistique de normalisation. Il vérifie les
empreintes des deux caches, le manifeste, les classes et l’égalité complète des
index : mêmes morceaux, fenêtres, positions, durées, partitions, artistes et
empreintes audio. Les doublons de fenêtres et toute ligne de test sont refusés.
Les variantes retirent des **colonnes**, sans retirer de morceaux ou de fenêtres.

La clé de chaque cache inclut les rapports parents, leurs configurations complètes,
les colonnes sélectionnées, la version de NumPy et les empreintes du code.
Une nouvelle exécution avec les mêmes parents et le même code réutilise le cache
vérifié sans changer son rapport ni casser les références des expériences passées.
Les artefacts altérés sont refusés ; les données audio originales restent intactes.

## Protocole d’apprentissage

Chaque variante utilise le script commun aux deux références : `StandardScaler`
ajusté sur les 18 897 fenêtres d’entraînement, puis régression logistique
multinomiale avec C dans {0,1 ; 1 ; 10}. Aucun rééquilibrage spécifique des familles
ni poids ajusté sur validation n’est ajouté. Le scaler standardise chaque dimension,
ce qui n’équilibre pas le poids global de familles de tailles différentes.

À l’évaluation, les probabilités des fenêtres sont moyennées pour chaque morceau.
C maximise la macro-F1 sur validation ; les égalités favorisent le plus petit C.
Les trois variantes disposent du même budget de recherche. Le rapport présente
toute la grille, ce qui permet aussi la comparaison à C = 0,1 fixe.

Les scalers et coefficients sont exportés en tableaux NumPy sans pickle. Après
rechargement, leurs probabilités sont vérifiées sur toutes les fenêtres de
validation. La comparaison recalcule les métriques à partir des prédictions
exportées et vérifie que chaque bloc fusionné est identique à ses colonnes sources.

## Reproduction

Depuis la racine, avec les caches descripteurs et ViT déjà préparés :

```powershell
foreach ($fusionVariant in @('fusion', 'fusion_no_rhythm', 'fusion_no_harmony')) {
    .\.venv\Scripts\python.exe scripts/build_fusion_features.py --variant $fusionVariant
    if ($LASTEXITCODE -ne 0) { throw 'Échec du cache de fusion' }
    .\.venv\Scripts\python.exe scripts/train_descriptor_baseline.py --features $fusionVariant
    if ($LASTEXITCODE -ne 0) { throw 'Échec de l’entraînement' }
}
.\.venv\Scripts\python.exe scripts/compare_fusion_baselines.py
```

Configurations : `configs/baseline_fusion*.json`. Pointeurs des caches :
`data/features/latest_fusion*.json`. Pointeurs des expériences :
`reports/latest_fusion*_run.json`. Chaque rapport de cache contient le prétraitement
des deux parents et l’ordre exact des caractéristiques nécessaires à l’inférence.

## Interprétation et suite

La fusion complète corrige 50 erreurs du ViT, mais perd 69 de ses bonnes prédictions.
La complémentarité observée entre deux systèmes séparés ne suffit donc pas à
garantir une meilleure combinaison. Le retrait d’une famille modifie aussi la
dimension et la régularisation effective du modèle ; il ne démontre pas que le
rythme ou l’harmonie sont intrinsèquement nuisibles.

Le ViT seul reste la meilleure référence actuelle selon la macro-F1 de validation.
Les écarts, notamment les 0,0060 entre ViT et fusion sans rythme, ne sont pas
accompagnés d’une preuve de significativité. La validation a servi au développement ;
le test n’est pas évalué. Prochaine étape : benchmark puis référence CNN prévue au
plan, avant les autres ablations et le verrouillage du protocole final.
