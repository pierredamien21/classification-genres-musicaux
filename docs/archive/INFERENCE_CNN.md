# Inférence des trois références CNN

État au 29 octobre 2026 : T5 vérifiée, 207 tests réussis.
[Résultats et preuves](../../reports/INFERENCE_CNN.md).

`CnnPredictor` charge un export sélectionné de `SmallAudioCNN`, sans entraînement
et sans sélection supplémentaire. Graines autorisées : 42, 43 et 44. Les époques
épinglées sont respectivement 15, 10 et 8.

## Utilisation

```python
from pathlib import Path
from music_genre.inference.cnn_adapter import CnnPredictor

root = Path("E:/projet/vision transformer")
predictor = CnnPredictor(root, seed=42)
# Remplacer par un fichier WAV/MP3 existant.
result = predictor.predict_file(root / "data/interim/mon_extrait.wav", mode="fma")
print(result["top3"])
```

Le contrat audio et la sortie sont ceux de [l’API du candidat](../INFERENCE_CANDIDAT.md) :
fichier borné avant décodage, refus du silence, fenêtres FMA 0/10/20 secondes,
complément à droite, omission des positions hors signal, scores par fenêtre et
moyenne par morceau. L’API ne reçoit aucune étiquette ni identité d’artiste.

## Prétraitement et calcul

Le signal mono 16 kHz produit un log-Mel natif de 128 × 622, un seul canal.
La normalisation CNN conserve les bornes −80/+20 dB et la transformation vers
[−1, 1]. Aucun redimensionnement 224 × 224 ni normalisation ImageNet n’intervient.
L’extraction réutilise `log_mel`, le réseau `SmallAudioCNN` et l’inférence
historique `predict_windows`, avec mémoire `channels_last` et poids gelés.

Les probabilités de fenêtre produites en float32 sont converties en float64
avant la moyenne, comme dans l’agrégation historique. Un fichier est traité
avec ses seules fenêtres : jusqu’à trois, contre des lots de seize lors du
calcul de validation complet. Les contrôles enregistrent aussi les différences
par fenêtre, sans supposer que la taille du lot est toujours neutre numériquement.

Les spectrogrammes sont extraits avec un thread numérique ; la prédiction CNN
utilise les quatre threads annoncés par la configuration. Les réglages sont
restaurés après l’appel, même s’il échoue. Ils concernent le processus : des
appels concurrents dans un même processus n’ont pas été testés.

## Artefacts et vérifications

Les rapports de run et de cache, le poids `model.safetensors`, la graine, l’époque,
l’ordre des labels et la configuration sont contrôlés contre le recensement
préparatoire. L’architecture attendue et le chargement strict des poids empêchent
de charger silencieusement une autre forme de réseau. L’API n’a pas besoin des
matrices de cache ou des labels de validation ; le vérificateur les utilise.

```powershell
.\.venv\Scripts\python.exe scripts/check_cnn_inference.py
.\.venv\Scripts\python.exe scripts/verify_cnn_inference.py
```

Le premier script recalcule les prédictions de validation depuis le cache natif,
puis vérifie les 64 fichiers du panel pour chaque graine. Le second contrôle les
empreintes, les moyennes des fenêtres, l’identité des classes prédites, les matrices
de confusion et la moyenne/dispersion des métriques. L’état réel de la campagne
est dans `reports/CNN_INFERENCE_CHECK.json` et sa progression dans
`reports/CNN_INFERENCE_PROGRESS.json`.

Les trois réseaux restent trois répétitions du CNN de référence. Leur synthèse
est une moyenne de métriques, pas une moyenne des probabilités. La CLI, la démo
et un paquet de déploiement autonome restent à réaliser.
