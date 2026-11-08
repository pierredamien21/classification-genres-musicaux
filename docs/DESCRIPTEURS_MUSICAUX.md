# Descripteurs musicaux : ce que mesure la première référence

L’objectif est de disposer d’une référence interprétable avant de mesurer l’apport
du Vision Transformer. Chaque fenêtre devient un vecteur de **125 valeurs**.
Ce ne sont pas des étiquettes de genre calculées à la main : le classifieur apprend
les relations entre ces mesures et les huit genres à partir de training.

## Les trois familles

| Famille | Mesures | Dimensions | Interprétation et limites |
|---|---|---:|---|
| Timbre / énergie | Moyenne et écart-type de 20 MFCC | 40 | Forme de l’enveloppe spectrale ; information locale, pas reconnaissance d’instruments |
| Timbre / énergie | Centroïde et largeur spectrale | 4 | Répartition des fréquences et dispersion, en Hz |
| Timbre / énergie | Contraste spectral, sept bandes | 14 | Différences entre pics et creux spectraux |
| Timbre / énergie | RMS | 2 | Énergie du signal, sensible au niveau d’enregistrement |
| Harmonie | Chroma à douze classes de hauteur | 24 | Répartition des hauteurs ramenées à une octave ; perd le registre et ne transcrit pas les accords |
| Rythme | Autocorrélation du flux spectral | 41 | Périodicités d’attaques, évaluées sur 40–240 BPM par pas de cinq |

Les statistiques moyennes/écarts-types résument l’évolution dans une fenêtre, mais
perdent l’ordre précis des événements. Cette perte est une limite mesurable de la
référence, pas une preuve anticipée que le Transformer fera mieux.

Le profil rythmique conserve plusieurs pics. Un motif à 120 pulsations par minute
peut aussi produire une réponse à 60 : il ne faut pas lire chaque pic comme un tempo
certain. Le profil utilise une autocorrélation centrée, normalisée par son énergie
au retard zéro ; une enveloppe constante produit un profil nul.

## Prétraitement figé pour cette expérience

- Décodage float32, moyenne des canaux, rééchantillonnage soxr_hq à 16 kHz.
- Fenêtres commençant à 0, 10 et 20 s, durée maximale de dix secondes.
- STFT Hann, FFT 1 024, pas 256, sans centrage.
- Mel Slaney 128 bandes, de 0 à 8 kHz ; puissance en dB avec référence 1,
  plancher `1e-10`, sans écrêtage relatif `top_db`.
- MFCC par DCT-II orthonormale ; chroma issu de la puissance STFT,
  accordage fixé à zéro et normalisation par le maximum de chaque trame.
- Flux d’attaques : différences positives du log-Mel, moyennées sur les bandes.
  L’autocorrélation est interpolée aux retards correspondant à la grille BPM.

Les descripteurs utilisent uniquement les trames complètes de la portion audio réelle.
Pour un signal plus court que la FFT, une trame est complétée par des zéros.
Cela évite de transformer le padding final en silence artificiel dans les statistiques.
Le début, la durée réellement disponible et le padding restent inscrits dans l’index.

Le niveau audio n’est pas normalisé morceau par morceau et les amplitudes float
supérieures à un ne sont pas coupées. RMS et le premier MFCC peuvent donc capturer
des différences de niveau/mastering. Une ablation de ces informations sera une
question expérimentale possible ; ce premier modèle n’isole pas leur contribution.

## Protocole d’apprentissage

La première extraction charge **train et validation seulement**. Les sources de test
ne sont pas décodées par cette étape. Chaque fenêtre de training reçoit le genre
du morceau. StandardScaler apprend ses moyennes et échelles sur ces seules fenêtres.

Trois régressions logistiques multinomiales sont ajustées, avec C = 0,1 / 1 / 10.
Pour chacune, les probabilités des fenêtres de validation sont moyennées par morceau.
Le C retenu maximise la macro-F1 par morceau ; en cas d’égalité, le plus petit C gagne.
Le prédicteur majoritaire utilise les effectifs des morceaux de training.

La validation ayant servi à la sélection, elle n’est pas une estimation finale indépendante.
Le test et ses intervalles de confiance restent réservés au verrouillage du protocole.
Les résultats concernent la sélection filtrée décrite dans `reports/REVUE_DONNEES.md`.

## Exécuter et reprendre

```powershell
.\.venv\Scripts\python.exe scripts/extract_descriptors.py --benchmark
.\.venv\Scripts\python.exe scripts/extract_descriptors.py --workers 4
.\.venv\Scripts\python.exe scripts/train_descriptor_baseline.py
.\.venv\Scripts\python.exe scripts/summarize_descriptor_baseline.py
```

Le benchmark mesure 64 fenêtres centrales, huit par genre, choisies dans training.
L’extraction exige un benchmark correspondant à la configuration et aux sources du code.
Le cache est identifié par une empreinte incluant manifeste, prétraitement, versions
des bibliothèques, ordre des classes et code d’extraction. Les fichiers audio sont
recontrôlés avant utilisation. Un échec interrompt le traitement ; aucun morceau
n’est supprimé silencieusement. Une relance réutilise les fichiers par morceau déjà
valides dans ce cache.

`data/features/latest_descriptors.json` indique `features.npy`, `index.csv` et le rapport
de provenance. L’index explicite les clés `(track_id, window_id)`, l’ordre des lignes
et les partitions. Les tableaux ne contiennent pas d’objets Python sérialisés.

`reports/latest_descriptor_run.json` indique l’expérience dans `runs/`, avec
configuration, prédictions de validation, statistiques des caractéristiques sur train
et export numérique du classifieur. `model.npz` contient uniquement le scaler,
les coefficients, les biais et l’ordre des classes ; `model.json` complète le schéma.
Le rechargement est comparé aux probabilités scikit-learn sur toute la validation.
La démo audio et la CLI de prédiction restent à construire.

## Sources de mise en œuvre

[Fonctions spectrales de librosa](https://librosa.org/doc/main/_modules/librosa/feature/spectral.html)
pour les descripteurs ; [prévention des fuites de données dans scikit-learn](https://scikit-learn.org/stable/common_pitfalls)
pour l’ajustement des transformations uniquement sur training.
Le schéma expérimental général et ses lectures sont dans `docs/REVUE_BIBLIOGRAPHIQUE.md`.
