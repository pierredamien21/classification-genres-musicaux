# Référence CNN entraînée de zéro

Expérience terminée : **macro-F1 0,5301**, accuracy **0,5577** sur les 780 morceaux
de validation, à l’époque 9 parmi dix époques effectuées. Le ViT reste premier selon
la macro-F1 (0,5447), malgré une accuracy légèrement inférieure (0,5538).
Voir [le rapport et l’historique complet](../../reports/BASELINE_CNN.md).

## Question et choix fixés avant les résultats

Cette référence situe le transfert du ViT par rapport à un petit réseau appris
directement sur les log-Mel de FMA. Les morceaux et fenêtres sont identiques.
Le CNN conserve la résolution native **128 bandes × 622 trames** ; le ViT reçoit
une image redimensionnée en 224 × 224 et bénéficie d’un préentraînement image.
Cette comparaison ne permet donc pas d’attribuer un écart à la seule architecture.

La configuration initiale `configs/cnn.json` fixe :

- Trois blocs convolution 3 × 3, ReLU et max-pooling 2 × 2 : 16, 32 puis 64 canaux.
- Pas spatial (stride) de 2 dans la première convolution, puis 1 dans les suivantes, padding 1.
  Ce premier sous-échantillonnage limite le calcul CPU ; il est décidé avant
  l’observation des résultats de validation.
- Pooling moyen global et tête linéaire de huit classes : **23 816 paramètres**.
- Pas de poids préentraînés, de BatchNorm, de dropout ou d’augmentation.
- Plage fixe [-80,20] dB transformée en [-1,1], sans statistiques apprises sur
  validation. Même FFT, hop, bandes Mel, fenêtres et padding audio que le ViT.
- AdamW, taux d’apprentissage 0,001, weight decay 0,0001, batch de 16, quatre fils CPU.
- Graine 42, dix époques maximum, arrêt après trois époques sans hausse de la
  macro-F1 par morceau sur validation. À égalité, conserver l’époque antérieure.

Une époque parcourt toutes les fenêtres de train dans un ordre mélangé
reproductible. La cible de chaque fenêtre est le genre du morceau. Les probabilités
sont moyennées par morceau avant les métriques de validation. Aucune recherche
additionnelle de taux d’apprentissage ou d’architecture n’est incluse dans cette
première expérience. Les trois graines finales prévues au plan restent distinctes.

Les champs descriptifs d’architecture dans la configuration documentent le modèle
v1 codé dans `SmallAudioCNN` ; changer l’architecture exige aussi de changer le code.

## Faisabilité mesurée

`scripts/benchmark_cnn.py` sélectionne 64 fenêtres centrales de train, huit par
genre, graine 42. Il mesure décodage/log-Mel, puis de vrais pas avant/arrière avec
mise à jour AdamW, après une chauffe exclue. Aucun modèle du benchmark n’est utilisé
comme initialisation de l’expérience complète.

Mesure initiale : environ **61,5 secondes de calcul par époque**, soit 10,3 minutes
pour dix époques, hors lectures du cache, métriques, sauvegardes et extraction.
La mémoire RSS maximale observée aux points d’échantillonnage est de 593 MiB ;
ce n’est pas une mesure exhaustive du pic mémoire. Rapport :
[`CNN_BENCHMARK.json`](../../reports/CNN_BENCHMARK.json).

Temps effectivement observés : extraction du cache 3,72 minutes ; dix époques avec
validation 11,60 minutes ; 12,07 minutes en incluant export et évaluation du train.
L’arrêt est dû au maximum de dix époques fixé avant résultats, pas au plafond de
douze heures ni à une convergence établie.

Le plafond reste douze heures par expérience. Le script vérifie ce budget en cours
d’entraînement et conserve explicitement un motif d’arrêt. Une époque incomplète
à l’épuisement du budget est abandonnée au profit du dernier checkpoint complet.
L’évaluation et l’export final peuvent ajouter un petit coût au temps d’entraînement.

## Cache sur disque

`scripts/extract_cnn_logmel.py` filtre train/validation avant ouverture des MP3.
Il contrôle les empreintes audio, puis vérifie les positions et durées réelles
contre l’index des descripteurs. Les log-Mel float32 ne sont ni redimensionnés ni
quantifiés. L’index final est copié octet pour octet depuis la référence vérifiée.

Le cache contient une matrice de **21 237 × 128 × 622 float32**, environ 6,76 Go
(6,30 Gio). Les fichiers par morceau, conservés pour reprise, occupent un volume
similaire : environ 13,5 Go au total, hors petits fichiers de métadonnées. Le script
contrôle l’espace disque avant extraction. Les batches lisent une matrice mappée
sur disque ; aucun chargement intégral en RAM n’est requis.

La clé du cache dépend du manifeste, des paramètres audio, du rapport de référence,
des versions et des sources. Les fichiers par morceau et la matrice assemblée ont
leurs empreintes SHA-256. Une extraction complète déjà vérifiée peut être réutilisée.

## Checkpoints, sélection et export

Après chaque époque complète, `checkpoint.pt` est remplacé atomiquement. Il contient
les poids courants et les meilleurs poids, l’état AdamW, l’état aléatoire, l’historique,
le temps dépensé et l’identité de la configuration, des données et du code. Le
chargement utilise `weights_only=True` et refuse une identité différente.

Une interruption au milieu d’une époque repart du dernier point complet. Une
interruption forcée du processus peut perdre le temps non sauvegardé depuis ce
point ; l’arrêt normal sur budget comptabilise aussi l’époque abandonnée. Les
checkpoints ne changent pas les partitions ou le budget expérimental annoncé.

L’export d’inférence utilise `model.safetensors`, accompagné de `model.json` avec
prétraitement et ordre des classes. Les probabilités du modèle exporté sont
comparées à celles du meilleur état en mémoire sur toute la validation. Le score
train final est calculé avec ce même meilleur état en mode évaluation.

## Commandes

Depuis la racine, dans l’environnement local vérifié :

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_cnn.py
.\.venv\Scripts\python.exe scripts/extract_cnn_logmel.py --workers 4
.\.venv\Scripts\python.exe scripts/train_cnn.py
.\.venv\Scripts\python.exe scripts/summarize_cnn.py
```

Reprise d’une expérience interrompue, sans modifier code, configuration ou caches :

```powershell
.\.venv\Scripts\python.exe scripts/train_cnn.py --resume runs/cnn-IDENTIFIANT
```

`reports/latest_cnn_training.json` indique le dossier en cours, dont `progress.json`
décrit l’état. `reports/latest_cnn_run.json` n’est publié qu’après l’évaluation et
l’export vérifiés. Une exécution du benchmark modifie son rapport daté par ses
mesures : ne pas le relancer pendant une expérience à reprendre, dont l’identité
inclut l’empreinte de ce rapport.

Les résultats de validation servent à sélectionner l’époque et comparer les modèles.
Ils ne constituent pas une estimation finale indépendante. Le test reste réservé
à la procédure finale, après choix des configurations et répétitions prévues.
