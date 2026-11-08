# Inférence commune du candidat bloc 6 + HPSS

La classe `CandidatePredictor` charge une fois les poids et le classificateur,
puis accepte un chemin WAV/MP3 sans label ni métadonnée d’artiste. Elle n’ajuste
aucun paramètre. La CLI utilisateur et la démo ne sont pas encore raccordées.

## Utilisation Python

```python
from pathlib import Path
from music_genre.inference.predict import CandidatePredictor

root = Path("E:/projet/vision transformer")
predictor = CandidatePredictor(root)
# Remplacer ce chemin par un fichier personnel WAV/MP3 existant.
result = predictor.predict_file(root / "data/interim/mon_extrait.wav", mode="fma")
print(result["top3"])
```

Le mode `fma` analyse les positions 0, 10 et 20 secondes présentes dans le fichier,
avec dix secondes par fenêtre. Il n’analyse donc pas tout un long morceau.
Pour un extrait court, les positions hors du signal sont omises ; la dernière
fenêtre est complétée à droite. Aucun vote n’est créé par duplication. Le mode
début/milieu/fin de la future démo n’est pas encore implémenté ; tout autre nom
de mode est refusé.

La représentation et les normalisations restent celles de l’expérience : mono
16 kHz, canaux original/harmonique/percussif, six blocs gelés, puis classificateur
linéaire exporté. Les scores par morceau sont la moyenne des probabilités des
fenêtres. Le résultat JSON contient labels ordonnés, scores, top-3, positions,
durées valides/padding, scores par fenêtre, métadonnées techniques d’entrée,
temps de calcul et `model_id` (empreinte du classificateur, des labels et de la
configuration de représentation). Le temps retourné exclut le chargement initial.
Ces scores ne sont pas calibrés comme des probabilités de vérité.

## Pièces nécessaires et contrôles

La racine contient le recensement `reports/PREPARATION_EVALUATION_FINALE.json`,
le rapport et `model.npz` du run candidat, son rapport de caractéristiques et
les poids préentraînés épinglés. Les matrices de caractéristiques, l’audio
d’apprentissage et les labels de validation ne sont pas nécessaires à l’API.
Ce chargement exploite l’arborescence actuelle ; un paquet de déploiement
autonome reste à préparer pour la livraison.

Les rapports, modèle et poids sont vérifiés par SHA256 ; les chemins d’artefacts
restent dans leurs répertoires attendus. Les classes numériques doivent être
0 à 7 dans l’ordre, avec 192 caractéristiques. Une pièce manquante ou modifiée
fait échouer le chargement. Les empreintes assurent la cohérence avec le
recensement local de confiance, pas une authentification contre sa réécriture.

Avant décodage PCM : au plus 50 000 000 octets, 300 secondes, huit canaux,
et 128 Mio estimés pour le tableau float32 natif (une frame supplémentaire
incluse). Cette dernière borne concerne ce tableau, pas la mémoire totale du
processus PyTorch. Les fichiers dépassant ces limites sont refusés, sans
troncature silencieuse. Fichier vide, corrompu, données non finies ou signal
entièrement nul après conversion mono produisent une erreur explicite.
Le silence reste accepté dans le module de diagnostic historique ; le rejet
concerne cette nouvelle interface publique. Aucun détecteur de musique ou
seuil d’énergie appris n’est introduit.

## Vérifier la reproduction sur validation

```powershell
.\.venv\Scripts\python.exe scripts/check_candidate_inference.py
```

Ce script utilise exclusivement les 64 morceaux du panel de validation fixé
en partie 23. Il vérifie le manifeste, les empreintes audio et les prédictions
de référence, passe par l’API réelle, puis compare les probabilités par morceau
à 10⁻⁶ et les positions/durées des fenêtres à l’index historique. Il vérifie
aussi que les sources n’ont pas changé pendant le calcul. Aucun cache
d’embeddings ne remplace l’extraction audio dans ce contrôle.

Une exécution réussie écrit `reports/CANDIDATE_INFERENCE_CHECK.json` et son
pointeur avec empreinte ; un échec interrompt le script sans publier de nouveau
rapport de succès. Vérifier les empreintes de sources d’un ancien rapport
avant de le réutiliser. Ce contrôle ne mesure pas encore la sensibilité au
volume ou au bruit, et n’ouvre pas le test. Voir les tâches T2 et suivantes
dans [le suivi](../tasks/todo.md).
