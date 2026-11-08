# Artefacts des modèles

- `pretrained/` : poids externes et leur provenance (source, licence, révision, checksum).
- `checkpoints/` : états reprenables d’entraînement.
- `exports/` : modèles destinés à la prédiction.

Un export devra inclure poids/classificateur, scalers, configuration audio, ordre des classes, version de code, manifeste d’entraînement et provenance. Vérifier l’équivalence des prédictions avant/après export. Ne charger que des artefacts de confiance.

Les fichiers volumineux restent hors de Git. Aucun modèle n’est actuellement présent.

