# Vérification des comparateurs linéaires — T4

Mis à jour : 28 octobre 2026. Terminée : six systèmes, mêmes 64 morceaux de validation et 192 fenêtres par système, soit 1 152 vecteurs de probabilités. Aucun audio de test chargé.

## Résultats

| Système | Dimensions | Erreur maximale par morceau | Erreur maximale par fenêtre | Macro-F1 complète reproduite |
|---|---:|---:|---:|---:|
| descriptor | 125 | 2.22e-16 | 6.11e-16 | 0.4612 |
| vit | 192 | 3.33e-16 | 8.88e-16 | 0.5447 |
| fusion | 317 | 2.78e-16 | 8.88e-16 | 0.5219 |
| vit_block6 | 192 | 2.22e-16 | 7.22e-16 | 0.5780 |
| vit_hpss | 192 | 2.22e-16 | 5.27e-16 | 0.5844 |
| vit_block6_hpss | 192 | 2.22e-16 | 6.66e-16 | 0.5975 |

Tolérance conservée : 10⁻⁶, appliquée ici aux morceaux et aux fenêtres. Pour chaque système, les probabilités des 780 morceaux de validation sont aussi recalculées depuis les caches de caractéristiques et le modèle exporté ; matrices de confusion et métriques reproduisent les rapports historiques. Ce contrôle ne réextrait pas les 780 fichiers audio : l’inférence audio porte sur le panel fixé de 64 morceaux.

## Ajustements nécessaires au contrôle

- Les durées de padding historiques sont issues d’une soustraction, tandis que l’API divise un nombre entier d’échantillons par la fréquence. L’écart observé était inférieur à 8 × 10⁻¹⁶ seconde. La géométrie est maintenant comparée en nombres d’échantillons entiers, avec test de détection d’un décalage d’un échantillon.
- Le cas ViT `128484` avait un écart de fenêtre de 1,0868 × 10⁻⁶ avec huit threads BLAS, bien que l’écart par morceau restât sous 10⁻⁶. Le cache avait été extrait avec une limite d’un thread. Reproduire ce réglage annule l’écart sur ce cas ; un test de régression couvre l’appel depuis un contexte BLAS à huit threads.
- Les quatre nouvelles API limitent temporairement les threads pendant la prédiction ; le vérificateur applique aussi cette limite aux deux API historiques. Leurs sources et les preuves T2b/T3 restent inchangées. Aucun poids, scaler, paramètre acoustique ou seuil de conformité n’a changé.

## Fusion et artefacts

Les descripteurs et le ViT reçoivent le même objet fenêtre. Ordre ViT puis descripteurs, colonnes, noms et familles vérifiés contre le rapport épinglé ; aucune partition fictive injectée dans l’inférence. Le scaler chargé est celui du modèle de fusion. Les six index de caches correspondent exactement, et leurs identifiants, artistes, labels et empreintes audio sont contrôlés contre le manifeste.

Rapports, modèles, matrices, index, sources et poids sont contrôlés par SHA256. Les fichiers audio du panel sont vérifiés avant et après prédiction. Les nouvelles API fonctionnent depuis les rapports et exports ; les matrices servent au vérificateur, pas à la prédiction utilisateur.

## Vérifications et reproduction

196 tests passent et Ruff réussit. Les 18 nouveaux cas couvrent reproduction des quatre comparateurs, fusion mal configurée, artefacts altérés, fichier stéréo court, silence, géométrie en échantillons et contexte BLAS. La suite complète utilise un répertoire temporaire dans le workspace ; un essai ciblé sans ce réglage a rencontré la restriction du dossier temporaire Windows, puis la suite a été relancée avec succès.

Campagne complète : 237.22 secondes, sous le budget de trente minutes. Durée indicative : une partie a coïncidé avec les tests, ce n’est pas un benchmark comparatif de vitesse.

Le vérificateur secondaire recalcule les moyennes des fenêtres, les écarts aux références et les métriques par matrices de confusion. Il confirme 1 152 vecteurs de probabilités et six recalculs sur 780 morceaux, avec contrôle des empreintes.

```powershell
.\.venv\Scripts\python.exe scripts/check_comparator_inference.py
.\.venv\Scripts\python.exe scripts/verify_comparator_inference.py
```

[Résultats et provenance](COMPARATOR_INFERENCE_CHECK.json) · [Recalcul](COMPARATOR_INFERENCE_RECOUNT.json) · [API et limites](../docs/INFERENCE_COMPARATEURS.md)

## Suite

T4 est satisfaite. Poursuivre T5 avec les trois CNN sélectionnés (graines 42/43/44), leur prétraitement natif et leurs exports ; aucun nouvel entraînement. Le verrouillage T6 et l’évaluation test T7 restent à faire. Les limites scientifiques précédentes, dont la sensibilité au bruit et la sélection sur validation, sont conservées.
