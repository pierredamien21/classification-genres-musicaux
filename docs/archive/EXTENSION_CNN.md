# CNN : prolongation contrôlée jusqu’à vingt époques

Protocole fixé le 21 octobre 2026 avant les nouvelles époques. Résultats initiaux :
[trois graines à dix époques](../../reports/REPETITIONS_CNN.md).

## Question et comparaison

Les époques choisies 9/10/8 sur dix suggèrent de vérifier si le plafond initial
limite cette recette. Comparer les plafonds dix et vingt sur les **mêmes graines
42/43/44**, sans changer les données, le CNN, les lots, l’optimiseur, le taux
d’apprentissage, la normalisation ni la patience de trois époques.

Chaque prolongation commence au dernier checkpoint de l’époque 10, dans un
nouveau dossier. Elle conserve poids courants, moments AdamW, RNG, historique,
meilleurs poids et compteur de patience. Les poids courants peuvent différer
des meilleurs poids exportés : repartir de ces derniers modifierait l’expérience.
Les parents, leurs rapports et leurs pointeurs restent conservés.

Sélection : meilleure macro-F1 de validation par morceau parmi **toutes** les
époques terminées, égalité départagée par la première époque. Patience inchangée,
donc vingt est un plafond et non un nombre imposé. Trois fenêtres par morceau,
moyenne de leurs probabilités. Test non chargé et non évalué.

## Coût et décision

Estimation avant lancement : environ **69 minutes supplémentaires au total**,
obtenues avec la plus lente époque observée de chaque graine, dix époques
supplémentaires, une marge de 50 % et 120 secondes d’export par run. C’est une
estimation, pas une garantie. La limite totale de douze heures par expérience
reste inchangée et inclut le temps d’apprentissage hérité du parent.

Rapporter chaque résultat, moyenne et écart-type d’échantillon entre graines,
ainsi que les différences appariées avec dix époques. Aucun choix de la meilleure
graine ni ensemble. Une hausse sur validation ne suffit pas à conclure à une
meilleure généralisation : le nombre d’époques candidates augmente et l’extension
a été décidée en observant cette validation. Les résultats négatifs seront conservés.

## Traçabilité et commandes

`configs/cnn_extension.json` fige les parents et leurs checkpoints par SHA-256,
les sources, les graines et le budget. La préparation refuse de remplacer ce
fichier. `scripts/train_cnn.py` reste inchangé ; la boucle d’apprentissage,
sélection et export du nouveau script est contrôlée à l’identique par un test.
Le test de continuation contrôle aussi poids, optimiseur et RNG contre une
exécution sans interruption. Un changement de source ou de configuration refuse
la reprise.

```powershell
# Protocole déjà créé : ne pas relancer pour l’écraser.
.\.venv\Scripts\python.exe scripts/prepare_cnn_extension.py
.\.venv\Scripts\python.exe scripts/train_cnn_extended.py --seed 42
.\.venv\Scripts\python.exe scripts/train_cnn_extended.py --seed 43
.\.venv\Scripts\python.exe scripts/train_cnn_extended.py --seed 44
.\.venv\Scripts\python.exe scripts/summarize_cnn_extension.py
.\.venv\Scripts\python.exe scripts/summarize_validation.py
```

Pour reprendre une prolongation interrompue : même graine et
`--resume runs/<dossier_cnn20>`. Les pointeurs
`reports/latest_cnn20_seed42_training.json` et équivalents donnent les dossiers ;
`latest_cnn20_seed42_run.json` n’est publié qu’après export vérifié.
Les temps `additional_training_validation_seconds` et
`inherited_training_validation_seconds` du rapport décrivent l’invocation courante
et son checkpoint de départ ; après plusieurs reprises, le temps total se lit
dans `training_validation_seconds`.

Les trois prolongations sont terminées et leur [synthèse vérifiée](../../reports/EXTENSION_CNN.md)
est publiée : macro-F1 moyenne 0,5407 ± 0,0105, gain moyen +0,0054 porté par la
seule graine 42. Arrêts anticipés aux époques 18/13/11, époques retenues 15/10/8.
Le jeu de test reste réservé à la version finalisée.
