# Référence ViT gelée

## Question expérimentale

Un ViT préentraîné sur des images fournit-il une représentation exploitable des genres
musicaux à partir de spectrogrammes ? La comparaison utilise les mêmes morceaux,
fenêtres, labels, partitions et classifieurs que la référence par descripteurs.
Elle compare deux représentations complètes ; elle n’isole pas l’effet de
l’architecture, du préentraînement ou du nombre de dimensions.

## Signal, image et vecteur

1. Décoder le MP3, moyenner les canaux et rééchantillonner à 16 kHz avec `soxr_hq`.
2. Découper les fenêtres [0,10], [10,20] et [20,30] secondes. Compléter la fin par
   des zéros si nécessaire ; conserver la durée réellement observée dans l’index.
3. Calculer un spectrogramme de puissance : fenêtre Hann, FFT 1 024, pas 256,
   sans centrage, 128 bandes Mel Slaney entre 0 et 8 kHz. Conversion dB avec
   référence 1, plancher numérique 1e-10, sans `top_db` relatif au morceau.
4. Appliquer la plage fixe [-80, 20] dB, puis borner à [0,1]. Cette plage reprend
   le diagnostic initial ; elle n’est ni ajustée sur validation ni réputée optimale.
   Il n’y a pas de normalisation min/max propre à chaque image.
5. Redimensionner en 224 × 224 par interpolation bicubique avec antialiasing,
   reborner à [0,1], répéter sur trois canaux et appliquer la moyenne et l’écart-type
   du checkpoint : 0,5 pour chaque canal. Aucune palette de couleurs ni fichier PNG
   intermédiaire n’est utilisé.
6. Utiliser `vit_tiny_patch16_224.augreg_in21k_ft_in1k` en mode évaluation,
   avec gradients désactivés. Exclure les tokens de préfixe et moyenner les
   196 tokens de patches du dernier bloc : **192 caractéristiques par fenêtre**.

La révision et les empreintes du checkpoint sont copiées dans
`configs/vit_frozen.json`. Les fichiers locaux sont contrôlés avant chargement ;
aucun téléchargement implicite n’est effectué par l’extraction.

La répétition sur trois canaux adapte l’entrée au réseau image ; elle ne crée pas
trois modalités audio. Le redimensionnement carré réduit la résolution temporelle.
La moyenne des tokens conserve une représentation globale, sans transcription des
paroles, des instruments ou des accords. Contrairement aux statistiques des
descripteurs, le ViT reçoit la fenêtre complétée entière : les positions et durées
restent identiques, mais la gestion du padding dans les représentations diffère.

## Cache et séparation des données

`scripts/extract_vit.py` filtre le manifeste sur `train` et `validation` avant
toute ouverture audio. La clé du cache comprend configuration, checkpoint,
empreinte du manifeste, versions des bibliothèques et empreintes des sources.
Chaque MP3 est vérifié par SHA-256. Les fichiers par morceau sont enregistrés
par remplacement atomique et permettent de reprendre une extraction interrompue.

La matrice finale contient uniquement les vecteurs, pas les spectrogrammes.
L’index conserve identifiant du morceau, artiste, partition, label, position,
durée valide, padding et empreinte audio. L’entraînement ViT exige que cet index
soit identique octet pour octet à celui des descripteurs. Un changement des
données ou du prétraitement impose donc une nouvelle comparaison cohérente.

## Classifieur et lecture des résultats

Le script commun `train_descriptor_baseline.py --features vit` ajuste un
`StandardScaler` sur les fenêtres d’entraînement puis une régression logistique
multinomiale. La grille C = {0,1 ; 1 ; 10}, le solveur et la tolérance sont les mêmes
que pour les descripteurs. Les probabilités des fenêtres sont moyennées par morceau.
Le choix maximise la macro-F1 sur validation ; à égalité, le plus petit C gagne.

Le réseau ViT n’est pas réentraîné. Les scores publiés concernent la validation
utilisée pour sélectionner C, sur FMA-small filtré ; ils ne sont pas des scores
finaux sur test. Le prédicteur linéaire est exporté en tableaux numériques sans
pickle, puis ses probabilités sont comparées à celles de scikit-learn.

## Reproduction locale

Depuis la racine du projet, avec l’environnement documenté dans
[ENVIRONNEMENT.md](ENVIRONNEMENT.md) et les données préparées :

```powershell
.\.venv\Scripts\python.exe scripts/fetch_vit.py --verify-only
.\.venv\Scripts\python.exe scripts/extract_vit.py --benchmark
.\.venv\Scripts\python.exe scripts/extract_vit.py --workers 4
.\.venv\Scripts\python.exe scripts/train_descriptor_baseline.py --features vit
.\.venv\Scripts\python.exe scripts/compare_vit_baseline.py
```

Le benchmark mesure 64 fenêtres centrales de train, huit par genre, graine 42,
après une chauffe exclue de la médiane. L’extraction complète utilise quatre
processus et un fil numérique par processus. Les temps incluent le décodage et
les écritures ; une reprise depuis le cache sera plus rapide qu’un calcul initial.
Le benchmark doit être refait si la clé du cache change.

Résultats : [comparaison](../reports/COMPARAISON_VIT.md),
`reports/latest_vit_run.json` et `data/features/latest_vit.json`.
La prochaine expérience est la concaténation ViT + descripteurs avec la même
grille, sans présumer un gain. Le CNN, les ablations, le verrouillage final et la
démo restent des étapes distinctes.
