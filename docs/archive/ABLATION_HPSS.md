# Ablation des canaux original, harmonique et percussif

Résultat vérifié sur les 780 morceaux de validation : **macro-F1 0,5844,
accuracy 0,5974**, avec C = 1, contre **0,5447 / 0,5538** pour la référence
bloc 12. Le gain de macro-F1 est de **0,0397**. Cette variante devient le meilleur
résultat de développement, légèrement devant le bloc 6 à 0,5780 ; cet écart
ne constitue pas une preuve de supériorité sur de nouveaux artistes.
Les F1 de sept genres augmentent contre la référence bloc 12 ; Hip-Hop diminue
de 0,7343 à 0,6927 et Pop reste faible à 0,1633. Voir le
[rapport détaillé](../../reports/ABLATION_HPSS.md) et la
[synthèse de toutes les expériences](../../reports/SYNTHESE_VALIDATION.md).

## Protocole fixé avant les résultats

L’hypothèse est que deux vues complémentaires du son peuvent aider le transfert
d’un ViT préentraîné sur des images. On compare les canaux **original/harmonique/
percussif** à trois copies du log-Mel original. Le réseau reste au **bloc 12** de
la référence : combiner cette modification au bloc 6 changerait deux facteurs.
L’ordre des canaux est fixé, sans recherche de permutation sur validation.

La séparation HPSS utilise le filtrage médian du spectre en fréquences linéaires.
Elle précède les filtres Mel et le passage en décibels. Paramètres explicites :
STFT de 1 024 échantillons, pas de 256, fenêtre Hann, sans centrage ; filtres
médians de 31 trames pour la composante harmonique et de 31 fréquences pour la
composante percussive ; puissance des masques 2, marges 1 et 1.

Les trois spectres complexes sont transformés séparément en puissance, puis en
128 bandes Mel Slaney et en dB avec référence fixe 1 et plancher 1e-10.
Aucune reconstruction audio intermédiaire ni normalisation indépendante du gain
des canaux n’est effectuée. Le canal original doit reproduire le log-Mel de la
référence. Chaque entrée a la forme 3 × 128 × 622, puis est redimensionnée à
224 × 224 avec les mêmes réglages bicubiques et la même plage fixe −80 à 20 dB.

La configuration conserve les trois fenêtres de dix secondes, les poids gelés,
la normalisation du checkpoint et la moyenne des tokens de patches après le
douzième bloc et sa LayerNorm. Le vecteur obtenu reste de dimension 192.
Standardisation sur train uniquement ; grille logistique C = 0,1 / 1 / 10 ;
sélection par macro-F1 de validation, égalité au plus petit C ; moyenne des
probabilités des fenêtres avant les métriques par morceau.

## Justification et limites de la séparation

La [documentation officielle de librosa](https://librosa.org/doc/0.11.0/generated/librosa.decompose.hpss.html)
décrit la séparation par filtrage médian et ses paramètres, en s’appuyant sur
Fitzgerald (DAFx 2010) et Driedger, Müller et Disch (ISMIR 2014). La signature et
l’implémentation de la version locale 1.0.0 ont également été inspectées.
Avec des marges de 1, les composantes complexes reconstruisent leur somme ;
leurs **puissances** ne sont pas une partition additive de l’énergie du mélange.
Cette méthode met en évidence des structures durables ou transitoires. Elle
n’identifie pas des instruments et ne garantit pas une séparation voix/batterie.

## Vérifications et traçabilité

Un benchmark préalable porte sur 64 fenêtres centrales de train, huit par genre,
graine 42, après une chauffe. Il vérifie le canal original contre l’ancien log-Mel
(tolérance 1e-5 dB) et les vecteurs issus de ce canal répété contre le cache ViT
initial (tolérance 1e-6). Ces vérifications ne sont pas incluses dans le temps
mesuré de la nouvelle représentation. Aucun score de validation n’intervient
dans les réglages HPSS.

Les MP3 sont redécodés : les log-Mel déjà en cache ne permettent pas de retrouver
le spectre linéaire requis pour HPSS. Les sources, versions, configurations,
checkpoint et référence sont inscrits dans la provenance. Le cache par morceau
est reprenable avec empreintes. Les index doivent être identiques à la référence.
Le jeu de test n’est pas extrait ni évalué pour cette étape.

## Diagnostic des entrées sur train

Le script `scripts/inspect_hpss.py` réutilise les 64 fenêtres de train du benchmark,
sans modifier les paramètres. Avant écrêtage et redimensionnement, les proportions
moyennes de valeurs hors de la plage fixe sont :

| Canal | Sous −80 dB | Au-dessus de +20 dB |
|---|---:|---:|
| Original | 0,214 % | 0,418 % |
| Harmonique | 0,542 % | 0,272 % |
| Percussif | 1,020 % | 0,035 % |

Ce diagnostic ne montre pas un écrêtage massif sur cet échantillon. Il ne mesure
ni la perte causée par le redimensionnement, ni la qualité de classification.
Les statistiques détaillées sont dans `reports/HPSS_INPUT_DIAGNOSTICS.json`.
L’exemple est le premier morceau Folk du benchmark fixe, FMA 132041, fenêtre
10–20 secondes. Les trois panneaux partagent la même échelle dB et sont affichés
avant redimensionnement ; aucune légende n’est fournie au modèle.

![Canaux HPSS sur un extrait de train](../../reports/figures/hpss_channels.png)

[Figure PDF exportable](../../reports/figures/hpss_channels.pdf).

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/extract_vit_hpss.py --benchmark
.\.venv\Scripts\python.exe scripts/extract_vit_hpss.py --workers 4
.\.venv\Scripts\python.exe scripts/train_descriptor_baseline.py --features vit_hpss
.\.venv\Scripts\python.exe scripts/compare_vit_hpss.py
.\.venv\Scripts\python.exe scripts/inspect_hpss.py
```

Configurations : `configs/vit_hpss.json`, `configs/baseline_vit_hpss.json`.
Pointeurs vérifiés : `data/features/latest_vit_hpss.json`,
`reports/latest_vit_hpss_run.json`. Cette étape consomme la deuxième et dernière
extraction supplémentaire du plan d’ablations. Les répétitions stochastiques
finales et le protocole de test restent à traiter après ces comparaisons.

Les 21 237 vecteurs sont extraits et les trois ajustements logistiques ont
convergé. Les 67 tests passent et l’export rechargé reproduit les probabilités
sans écart. Le benchmark mesure 0,4949 s par fenêtre pour HPSS/log-Mel/ViT.
La durée globale enregistrée de 311,32 minutes comporte de grands écarts entre
sauvegardes, détaillés dans le rapport ; elle n’est pas un benchmark de calcul
continu. Les figures utilisent Matplotlib, désormais inclus dans le relevé des
dépendances. La synthèse se reproduit avec `scripts/summarize_validation.py`.
