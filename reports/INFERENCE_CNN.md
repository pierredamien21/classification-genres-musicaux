# Vérification des trois checkpoints CNN — T5

Mis à jour : 29 octobre 2026. T5 terminée : les trois exports sélectionnés sont reproduits, sans nouvel entraînement ni accès à l’audio de test.

## Résultats

| Graine | Époque retenue | Macro-F1 validation | Accuracy validation | Erreur audio par morceau | Erreur audio par fenêtre | Erreur cache complet |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 15 | 0.5464 | 0.5615 | 0 | 0 | 0 |
| 43 | 10 | 0.5471 | 0.5590 | 0 | 0 | 0 |
| 44 | 8 | 0.5287 | 0.5474 | 0 | 0 | 0 |

Chaque réseau est vérifié sur les mêmes 64 fichiers audio et 192 fenêtres, soit 576 fenêtres au total. Les probabilités par fenêtre et morceau coïncident exactement avec les références lors de cette exécution. La tolérance annoncée reste 10⁻⁶ par morceau. Les métriques complètes sont reproduites depuis le cache natif pour les 780 morceaux de validation ; ces 780 fichiers audio n’ont pas été réextraits.

La macro-F1 moyenne des trois réseaux reste **0,5407 ± 0,0105**, et l’accuracy moyenne **0,5560 ± 0,0075** (écarts-types d’échantillon entre graines). Il s’agit d’une moyenne de métriques, sans ensemble de probabilités, sans meilleure graine choisie et sans nouvelle sélection d’époque. Ces scores de validation ne sont pas une estimation finale indépendante.

## Chaîne d’inférence conservée

`CnnPredictor` utilise le même contrat de fichier et de segmentation que le candidat : mono 16 kHz, fenêtres FMA 0/10/20 secondes, padding à droite, moyenne par morceau. Les spectrogrammes gardent leur résolution native 128 × 622 et leur canal unique. Les fonctions historiques `log_mel`, `normalize_logmel`, `make_batch` et `predict_windows` sont réutilisées ; aucun redimensionnement ViT ou normalisation ImageNet.

Poids `model.safetensors` chargés strictement dans `SmallAudioCNN`, mode évaluation, gradients désactivés. Les probabilités de fenêtre float32 passent en float64 avant la moyenne, comme dans `aggregate_tracks`. Extraction avec un thread, calcul CNN avec les quatre threads de sa configuration ; réglages restaurés après appel.

La prédiction d’un fichier utilise au plus trois fenêtres par lot, tandis que la validation complète utilise des lots de seize. Les écarts sont enregistrés explicitement : ils sont nuls sur ce panel, sans garantie générale pour tout matériel ou tout fichier.

## Preuves et limites

207 tests réussis, Ruff sans erreur. Les 11 nouveaux tests couvrent les trois graines, époques, reproduction audio, stéréo courte, silence, restauration des threads, artefacts modifiés, architecture incompatible, mauvais chemin et graine non prévue.

Le vérificateur secondaire contrôle les empreintes, les moyennes des 576 fenêtres, l’identité des 192 classes prédites (64 par graine), les matrices de confusion sur validation et les moyennes/écarts-types des métriques. Les sources historiques et les artefacts entraînés restent inchangés.

Campagne : 150.05 secondes, sous trente minutes. Cette durée inclut le contrôle des grandes matrices par empreinte ; elle n’est pas une mesure de latence de l’API.

Appels concurrents dans un même processus non testés. L’inférence exige encore l’arborescence locale des rapports et exports ; paquet autonome, CLI et démo restent à préparer. Les limites scientifiques précédentes (sélection sur validation, sensibilité au bruit) sont conservées.

```powershell
.\.venv\Scripts\python.exe scripts/check_cnn_inference.py
.\.venv\Scripts\python.exe scripts/verify_cnn_inference.py
```

[Résultats et provenance](CNN_INFERENCE_CHECK.json) · [Recalcul indépendant](CNN_INFERENCE_RECOUNT.json) · [API](../docs/INFERENCE_CNN.md)

## Prochaine action

T1 à T5 sont vérifiées ; le point B est satisfait. T6 doit maintenant produire un manifeste distinct, daté et contrôlé, avec refus des artefacts modifiés, partitions incohérentes ou protocole non verrouillé. Cette étape ne déclenche pas encore l’évaluation test T7. Le recensement préparatoire conserve son état historique non verrouillé.
