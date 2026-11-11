# Verrouillage final et coût prévisionnel — T6

29 octobre 2026. T6 terminée et vérifiée ; test non évalué.

## Périmètre

Neuf exports déjà sélectionnés ; 7 857 morceaux répartis en 6 299 train, 780 validation et 778 test. Le manifeste réunit 7 985 artefacts et 57 fichiers Python. Les preuves T1–T5, T2a/T2b et les quatre recomptages indépendants sont conservés avec leurs empreintes. Aucun nouvel entraînement.

[Manifeste daté](final-evaluation-lock-20261029T085227817846Z.json) — seul le pointeur `latest_final_evaluation_lock.json` publié après contrôle confirme sa validation.

## Coût mesuré sur train

Huit morceaux, un par genre, plus petit identifiant train ; trois fenêtres par fichier. Les prédictions ne servent à aucune sélection. Mesures séquentielles sur la machine locale, y compris le premier appel de chaque modèle ; BLAS et PyTorch à un thread pour les modèles linéaires, quatre threads PyTorch pour les CNN.

| Système | Secondes par morceau (moyenne) | Projection sur 778 morceaux (minutes) |
|---|---:|---:|
| descriptor | 0.440 | 5.7 |
| vit | 0.536 | 7.0 |
| fusion | 0.811 | 10.5 |
| vit_block6 | 0.400 | 5.2 |
| vit_hpss | 2.237 | 29.0 |
| vit_block6_hpss | 2.176 | 28.2 |
| cnn20_seed42 | 0.261 | 3.4 |
| cnn20_seed43 | 0.239 | 3.1 |
| cnn20_seed44 | 0.241 | 3.1 |

Total projeté d’inférence : **95.3 minutes**. Prévoir environ **1 h 35 à 3 h 10** avec une marge ×1 à ×2. Ce n’est pas un intervalle statistique : seulement huit morceaux ont été chronométrés, et charge machine, caches et chauffe peuvent modifier le débit. Cette durée exclut la répétition à blanc sur validation, les empreintes, le bootstrap et la rédaction des rapports.

## Vérifications et limites

24 tests du verrouillage passent ; 231 tests dans la suite complète. Ruff réussi. Les métadonnées réelles sont séparées par artiste, PCM et groupe de contenu. Les 7 985 empreintes ont été contrôlées avant et après le chronométrage. Relecture du manifeste et contrôle de la partition complète de validation réussis ; code de sortie 0. Durée totale : 240,66 secondes.

La configuration préparatoire reste inchangée et fermée ; le nouveau manifeste porte son propre statut. Son digest attendu doit être conservé dans le pointeur local. Ce dispositif détecte les modifications accidentelles, sans constituer une signature cryptographique contre une réécriture des deux fichiers.

Les fichiers test sont seulement hachés, sans décodage ni prédiction. La fragilité au bruit, les huit genres fermés, les scores non calibrés et l’absence de revue subjective demeurent des limites. T7 doit encore enregistrer son évaluateur et réussir une répétition sur validation avant la campagne finale.

## Exécution

Le premier lancement via `Start-Process` a disparu sans sortie. La session persistante 21880 a terminé avec succès le chronométrage puis la relecture des empreintes. La cause exacte de l’arrêt initial n’est pas établie. Références : `FINAL_LOCK_JOB.json` et `FINAL_LOCK_PROGRESS.json`.

[Procédure détaillée](../docs/archive/VERROUILLAGE_EVALUATION.md).

Empreinte du manifeste validé : `80eccf469a80dff691a02aaca20443eb7bbc6d952ab6516aca7e7a5c4ae446ec`. [Pointeur vérifié](latest_final_evaluation_lock.json).
