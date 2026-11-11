# Stabilité du CNN sur trois graines

## Protocole fixé avant les deux nouveaux résultats

Graines : **42, 43 et 44**, sans recherche de la meilleure graine. Le run de
développement 42 est réutilisé et identifié par son empreinte ; deux entraînements
supplémentaires sont terminés et vérifiés. Le [rapport](../../reports/REPETITIONS_CNN.md)
présente chaque score et la moyenne avec
l’écart-type d’échantillon (division par n − 1), sans remplacer la moyenne par
le meilleur run. Trois graines donnent une indication limitée de la variabilité,
pas un intervalle de confiance sur de nouveaux artistes.

Seules l’initialisation des poids et la permutation des lots changent. Même cache,
partitions, log-Mel, réseau, optimiseur, budget, patience et règle de sélection
d’époque. La limite reste dix époques, avec patience trois sur macro-F1 validation.
Des nombres d’époques différents sont donc possibles. Le benchmark de coût
initial reste applicable ; le changement de graine ne modifie ni taille ni calcul
par lot. Les répétitions sont séquentielles pour limiter la pression mémoire.

La validation choisit l’époque séparément dans chaque run. Les configurations
choisies ne sont pas réajustées après lecture des scores. Aucun ensemble des trois
réseaux n’est ajouté implicitement. Le test reste non évalué.

## Traçabilité et reprise

`configs/cnn_repetitions.json` fixe les graines, le run 42, les empreintes des
sources et le protocole statistique avant l’entraînement. Les sources originales
du CNN ont été vérifiées contre le run 42 avant modification. Le changement du
script ajoute seulement l’option de graine et des noms/pointeurs par répétition.
Le code des lots, de l’apprentissage et du modèle conserve ses empreintes.
Une copie du script initial est dans `artifacts/provenance/train_cnn_seed42_original.py`.

```powershell
.\.venv\Scripts\python.exe scripts/train_cnn.py --seed 43
.\.venv\Scripts\python.exe scripts/train_cnn.py --seed 44
.\.venv\Scripts\python.exe scripts/summarize_cnn_repetitions.py
```

Pour reprendre, ajouter `--resume runs/<dossier>` à la même commande `--seed`.
Les checkpoints gardent leur contrôle strict d’identité : une source ou une
configuration modifiée est refusée. Les pointeurs `latest_cnn_seed43_training.json`
et `latest_cnn_seed44_training.json` donnent les dossiers en cours. Les pointeurs
`latest_cnn_seed43_run.json` et `latest_cnn_seed44_run.json` ne sont publiés qu’après
export et vérification. La référence `latest_cnn_run.json` reste celle de la graine 42.

## Suite adaptative

Ces répétitions complètent la référence CNN, mais ne ferment pas l’exploration.
Les époques retenues 9/10/8 sur dix motivent de préparer un protocole à vingt
époques, mêmes graines et autres paramètres ; il n’est pas encore exécuté.
Un point de décision examinera aussi les erreurs Pop/Hip-Hop et l’intérêt
de nouvelles expériences (bloc 6 + HPSS, affinage ou modèle audio spécialisé).
Toute extension précisera son hypothèse, sa comparaison, son coût et sa règle de
décision avant exécution. Le verrouillage du test concerne une version choisie,
pas l’ensemble de la vie du projet.
