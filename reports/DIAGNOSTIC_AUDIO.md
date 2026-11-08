# Diagnostic technique des 19 extraits de revue

Recalcul depuis les MP3 originaux sur validation uniquement ; aucun entraînement ni changement de label.

## Correspondance avec les entrées et résultats enregistrés

| Piste | Genre FMA | Padding maximal (ms) | Erreur log-Mel max (dB) | Erreur embedding max | Erreur probabilité max |
|---|---|---:|---:|---:|---:|
| 064630 | Hip-Hop | 0.00 | 0 | 0 | 1.11e-16 |
| 075745 | Hip-Hop | 23.38 | 0 | 0 | 1.67e-16 |
| 005191 | Pop | 0.00 | 0 | 0 | 5.55e-17 |
| 070424 | Pop | 23.38 | 0 | 0 | 1.11e-16 |
| 012527 | Pop | 23.38 | 0 | 0 | 8.33e-17 |
| 089195 | Hip-Hop | 23.38 | 0 | 0 | 8.33e-17 |
| 113031 | Hip-Hop | 0.00 | 0 | 0 | 8.33e-17 |
| 051261 | Pop | 23.38 | 0 | 0 | 1.67e-16 |
| 104780 | Pop | 0.00 | 0 | 0 | 6.94e-17 |
| 045128 | Pop | 23.38 | 0 | 0 | 1.11e-16 |
| 086680 | Hip-Hop | 23.38 | 0 | 0 | 1.11e-16 |
| 070773 | Hip-Hop | 23.38 | 0 | 0 | 1.11e-16 |
| 014363 | Hip-Hop | 0.00 | 0 | 0 | 1.67e-16 |
| 009560 | Rock | 0.00 | 0 | 0 | 1.11e-16 |
| 148215 | Folk | 0.00 | 0 | 0 | 8.33e-17 |
| 126586 | Instrumental | 23.38 | 0 | 0 | 8.33e-17 |
| 120201 | Pop | 23.38 | 0 | 0 | 8.33e-17 |
| 128484 | Pop | 0.00 | 0 | 0 | 1.94e-16 |
| 061736 | Pop | 23.38 | 0 | 0 | 5.55e-17 |

## Prédictions HPSS selon la fenêtre

| Piste | Genre FMA | 0–10 s | 10–20 s | 20–30 s | Agrégation |
|---|---|---|---|---|---|
| 064630 | Hip-Hop | Hip-Hop | Electronic | Hip-Hop | Hip-Hop |
| 075745 | Hip-Hop | Hip-Hop | International | International | Hip-Hop |
| 005191 | Pop | Rock | Rock | Rock | Rock |
| 070424 | Pop | International | International | Pop | International |
| 012527 | Pop | Pop | Folk | Folk | Folk |
| 089195 | Hip-Hop | Experimental | Experimental | Pop | Experimental |
| 113031 | Hip-Hop | Experimental | Experimental | Hip-Hop | Experimental |
| 051261 | Pop | Experimental | Pop | Pop | Pop |
| 104780 | Pop | Pop | Pop | Electronic | Pop |
| 045128 | Pop | Pop | Pop | Pop | Pop |
| 086680 | Hip-Hop | Pop | Rock | International | Rock |
| 070773 | Hip-Hop | Electronic | International | Electronic | Electronic |
| 014363 | Hip-Hop | Hip-Hop | Electronic | Hip-Hop | Electronic |
| 009560 | Rock | Folk | Pop | Rock | Pop |
| 148215 | Folk | Rock | Pop | Pop | Pop |
| 126586 | Instrumental | Electronic | Pop | Pop | Pop |
| 120201 | Pop | Pop | Hip-Hop | Pop | Hip-Hop |
| 128484 | Pop | Electronic | Electronic | Pop | Electronic |
| 061736 | Pop | Experimental | Experimental | Experimental | Experimental |

## Figures représentatives

Une piste par catégorie, première dans la sélection préexistante. Ces figures révèlent les labels : les consulter après l’écoute indépendante.

- [064630 — hiphop_hpss_correction](../reports/figures/audio-diagnostic-20261025T231532303942Z/064630.png)
- [005191 — pop_rescued_block6](../reports/figures/audio-diagnostic-20261025T231532303942Z/005191.png)
- [089195 — hiphop_hpss_regression](../reports/figures/audio-diagnostic-20261025T231532303942Z/089195.png)
- [051261 — pop_hpss_correct](../reports/figures/audio-diagnostic-20261025T231532303942Z/051261.png)
- [086680 — hiphop_both_wrong](../reports/figures/audio-diagnostic-20261025T231532303942Z/086680.png)
- [009560 — pop_false_positive](../reports/figures/audio-diagnostic-20261025T231532303942Z/009560.png)
- [120201 — pop_shared_vit_error](../reports/figures/audio-diagnostic-20261025T231532303942Z/120201.png)

## Portée

Les fenêtres, spectrogrammes natifs, embeddings des trois ViT et probabilités par morceau ont été recalculés et comparés aux caches et prédictions enregistrés. Le canal original HPSS est comparé au log-Mel natif. Les empreintes des sources, versions, modèles, fichiers audio et caches sont contrôlées.

Cette reproduction détecte un écart avec le pipeline enregistré ; elle ne prouve pas que ses choix de mono, 16 kHz, fenêtres, redimensionnement ou préentraînement image sont optimaux. Un défaut partagé par extraction et recalcul peut subsister. Les trois CNN ne sont pas réexécutés : leur entrée log-Mel commune est contrôlée.

RMS calculé sur les seuls échantillons valides, sans padding ; les fractions hors plage dB quantifient l’écrêtage de la représentation, pas une saturation audible. Aucun genre ni instrument n’est identifié par écoute ici.

[Mesures par fenêtre, toutes les prédictions et empreintes](AUDIO_DIAGNOSTIC.json).

Commande : `.\.venv\Scripts\python.exe scripts/diagnose_review_audio.py`.
