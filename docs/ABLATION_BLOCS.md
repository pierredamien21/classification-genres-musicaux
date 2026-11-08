# Représentation ViT au bloc 6 ou au bloc 12

Résultat vérifié : **macro-F1 0,5780, accuracy 0,5885** au bloc 6, contre
**0,5447 / 0,5538** au bloc 12, sur les mêmes 780 morceaux de validation.
Le bloc 6 était le meilleur résultat à cette étape ; la variante HPSS au bloc 12
atteint depuis 0,5844, voir [l’ablation des canaux](ABLATION_HPSS.md). Les F1 par genre
augmentent dans cette expérience, mais Pop reste faible (0,1558).
Extraction complète : 3,97 minutes depuis les log-Mel en cache. Voir
[les résultats détaillés](../reports/ABLATION_BLOCS.md).

## Hypothèse et convention fixée avant les scores

Une représentation intermédiaire pourrait être moins spécialisée pour les classes
d’images du préentraînement et mieux transférer aux spectrogrammes. C’est une
hypothèse à vérifier ; aucun gain n’est présumé.

La variante utilise le même checkpoint ViT-Tiny gelé, le même redimensionnement,
les mêmes canaux, les mêmes paramètres log-Mel et les mêmes trois fenêtres que la
référence. Les six premiers blocs Transformer sont conservés ; les blocs 7 à 12
ne sont pas exécutés. Le calcul d’embeddings de patches, les positions et les
préfixes restent ceux du checkpoint.

Après le dernier bloc exécuté, on applique **la même LayerNorm finale gelée du
checkpoint**, puis on exclut les préfixes et on moyenne les tokens de patches.
Le vecteur reste de dimension 192. Cette LayerNorm a été apprise pour la sortie du
bloc 12 : son application au bloc 6 est une convention explicite. L’expérience ne
compare pas toutes les conventions de normalisation des représentations intermédiaires.

`FrozenViTBlock` hérite de l’extracteur existant et remplace seulement la séquence
de blocs par son préfixe. Le test vérifie la sortie du bloc 6 contre une capture
directe du sixième bloc du modèle complet, suivie de la LayerNorm. Le cas bloc 12
reproduit l’extracteur original. Les anciens caches et leur extracteur sont conservés.

## Entrées et traçabilité

L’extraction réutilise les log-Mel float32 du cache CNN, sans redécoder les MP3.
Avant tout calcul, les empreintes de la matrice, du rapport et de l’index sont
contrôlées. Les paramètres audio et log-Mel doivent être ceux de la référence ViT,
et les index des deux caches doivent être identiques octet pour octet.

Le benchmark porte sur 64 fenêtres centrales de train, huit par genre, graine 42.
Il mesure le bloc 6 après une chauffe et recalcule aussi le bloc 12 sur ces mêmes
entrées. L’écart maximal avec le cache ViT original doit être inférieur à 1e-6.
La mesure obtenue est de 0,0421 s par fenêtre au bloc 6, avec un écart bloc 12 nul.

Le cache du bloc 6 est reprenable par morceau. Sa clé inclut configuration,
checkpoint, rapports parents, versions des bibliothèques et empreintes des sources.
Les 21 237 fenêtres train/validation sont conservées ; aucune fenêtre de test
n’est chargée. Chaque fichier intermédiaire dispose de sa propre empreinte.

## Classifieur et comparaison

Le script commun ajuste un scaler sur train puis une régression logistique avec
C dans {0,1 ; 1 ; 10}. Sélection sur macro-F1 par morceau, égalité au plus petit C.
Les probabilités des trois fenêtres sont moyennées avant les métriques. Le ViT reste
gelé ; aucun paramètre de réseau ou de LayerNorm n’est réentraîné.

La comparaison vérifie la grille, les morceaux, l’index, les classes et la
configuration de prétraitement. Les seules différences de représentation prévues
sont la profondeur sélectionnée et son pooling documenté avec la même LayerNorm.
Les métriques sont recalculées depuis les prédictions sauvegardées et l’export
numérique du classifieur est rechargé pour vérifier ses probabilités.

## Reproduction

Depuis la racine, avec les caches ViT et log-Mel CNN disponibles :

```powershell
.\.venv\Scripts\python.exe scripts/extract_vit_block6.py --benchmark
.\.venv\Scripts\python.exe scripts/extract_vit_block6.py --workers 4
.\.venv\Scripts\python.exe scripts/train_descriptor_baseline.py --features vit_block6
.\.venv\Scripts\python.exe scripts/compare_vit_blocks.py
```

Configurations : `configs/vit_block6.json`, `configs/baseline_vit_block6.json`.
Pointeurs : `data/features/latest_vit_block6.json`, `reports/latest_vit_block6_run.json`.
Résultats : [rapport comparatif](../reports/ABLATION_BLOCS.md).

L’extraction sur log-Mel en cache n’inclut pas les coûts de décodage et de calcul
des spectrogrammes de l’expérience originale : les temps globaux ne sont donc pas
une mesure contrôlée de vitesse bloc 6/bloc 12. Cette étape consomme une des deux
extractions supplémentaires prévues pour les ablations. La seconde est réservée
à la variante original/harmonique/percussif sur le bloc 12, après benchmark.

La validation sert au développement. Le test et les répétitions stochastiques
finales restent réservés à la suite du protocole ; aucun écart significatif n’est
affirmé sur la seule base des scores de validation.
