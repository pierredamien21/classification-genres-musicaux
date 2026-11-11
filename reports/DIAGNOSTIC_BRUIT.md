# T2a — Diagnostic de la sensibilité au bruit

Diagnostic réalisé le 28 octobre 2026 sur les huit morceaux pilotes fixés avant
les essais. Aucun modèle, poids, seuil ou label modifié ; test non évalué.

## Conclusion permise par les mesures

Le bruit déplace les représentations standardisées du candidat beaucoup plus
que la baisse de volume, sans hausse massive des valeurs hors des bornes des
spectrogrammes. Les décisions initialement peu séparées sont davantage présentes
parmi les changements, mais ce phénomène ne suffit pas à les expliquer tous.
**Aucun bug de prétraitement n’est établi, et le rôle spécifique de HPSS reste
à distinguer de celui du transfert ViT et du classificateur.**

## Reproduction et portée

Les 72 fenêtres (8 morceaux × 3 conditions × 3 fenêtres) reproduisent exactement
les probabilités par fenêtre de T2 dans cette exécution : écart maximal zéro.
Les embeddings sauvegardés reproduisent également les scores et déplacements
de logits lors d’un rechargement indépendant. Durée relevée : 28,78 secondes,
sous le budget de dix minutes ; ce n’est pas une mesure comparative de vitesse.

Limite importante : sur les huit pilotes, l’identité donne quatre classifications
correctes, le gain trois, le bruit quatre. Les deux changements sous bruit sont
des passages entre deux classes incorrectes (Folk : Pop → Hip-Hop ; Rock :
Instrumental → Electronic). **Ce petit diagnostic ne reproduit donc pas les dix
régressions observées sur le panel complet.** Il localise des déplacements de
représentation sur un échantillon fixé et conserve les cas stables comme contrôles.

## 1. Bornes des spectrogrammes

Les mesures portent sur les log-Mel original/harmonique/percussif, avant
redimensionnement, bornés dans l’intervalle −80 à +20 dB. Elles ne quantifient
pas séparément le bornage après interpolation bicubique.

| Canal | Au-dessus de +20 dB, identité | Au-dessus de +20 dB, bruit | Sous −80 dB, identité | Sous −80 dB, bruit |
|---|---:|---:|---:|---:|
| Original | 0,4656 % | 0,4667 % | 0,1544 % | 0 % |
| Harmonique | 0,3135 % | 0,3135 % | 0,3179 % | 0,00335 % |
| Percussif | 0,04093 % | 0,04093 % | 0,6261 % | 0,00052 % |

Moyennes sur les 24 fenêtres de chaque condition. Le bruit fait principalement
sortir des valeurs de la zone sous la borne basse. L’hypothèse d’une apparition
massive de valeurs saturées aux bornes n’est pas soutenue ici ; cela n’exclut
pas tout effet local du bornage ou du redimensionnement.

## 2. Déplacements des entrées et des représentations

| Canal | Variation absolue moyenne log-Mel, gain | Variation absolue moyenne log-Mel, bruit | Variation entrée [0,1], gain | Variation entrée [0,1], bruit |
|---|---:|---:|---:|---:|
| Original | 5,9990 dB | 4,9092 dB | 0,05976 | 0,04897 |
| Harmonique | 5,9965 dB | 5,3357 dB | 0,05963 | 0,05308 |
| Percussif | 5,9948 dB | 6,4495 dB | 0,05943 | 0,06402 |

Les variations moyennes des entrées bornées sont voisines : 0,05961 pour le gain,
0,05535 pour le bruit. Leurs structures ne sont cependant pas équivalentes :
une amplitude moyenne de variation ne décrit pas les motifs temps/fréquence.

| Mesure, moyenne sur 24 fenêtres | Gain −6 dB | Bruit 20 dB |
|---|---:|---:|
| Distance L2 du déplacement après division par les échelles de train | 1,8943 | 12,9259 |
| Distance L2 relative à la norme de l’embedding original | 0,04522 | 0,29315 |
| Distance cosinus | 0,00108 | 0,04692 |

La première distance est **6,82 fois plus élevée sous bruit**. Il s’agit du
rapport de deux moyennes, pas d’un facteur garanti pour chaque morceau.
Le canal percussif varie davantage que les deux autres en moyenne sous bruit,
mais aucun canal n’a été remplacé isolément : on ne peut pas lui attribuer
causalement la dégradation. Les déplacements de logits moyennés sur les fenêtres
sont sauvegardés par classe ; les scores par morceau restent la moyenne des softmax,
et non le softmax de logits moyens.

## 3. Marges initiales sur le panel complet

Cette analyse réutilise uniquement les prédictions déjà calculées sur les
64 morceaux ; elle ne change pas les huit pilotes. La marge est l’écart entre
les deux scores les plus élevés avant bruit, pas une mesure calibrée de confiance.

| Groupe sous bruit | Morceaux | Marge initiale médiane | Min–max |
|---|---:|---:|---:|
| Classe changée | 26 | 0,2073 | 0,0032–0,8206 |
| Classe stable | 38 | 0,4612 | 0,0044–0,9121 |

Une marge faible est compatible avec une partie de la fragilité, mais les plages
se recouvrent et des marges initiales élevées n’empêchent pas certains changements.
C’est une association descriptive, sans démonstration causale ni significativité.

## Décision

Ne pas modifier la normalisation, ajouter un débruitage ou réentraîner sur la
base de ce seul diagnostic. Ajouter T2b : comparer **bloc 6 original**, déjà
entraîné, au candidat HPSS sur les mêmes 64 morceaux et les trois conditions.
Cela permettra de mesurer si la fragilité est plus marquée dans le système HPSS
ou partagée avec la référence à canaux répétés. Les classificateurs associés
ont été ajustés séparément sur train : la comparaison porte sur ces pipelines,
pas sur l’effet causal d’un canal isolé.

Le protocole T2b est précisé dans le suivi. Une correction
éventuelle devra ensuite faire l’objet d’une expérience distincte ; la décision
de verrouillage reste ouverte et le test fermé.

## Preuves et reproduction

- `scripts/diagnose_noise_sensitivity.py` avec `.venv\Scripts\python.exe`.
- [Rapport numérique et empreintes](NOISE_SENSITIVITY_DIAGNOSTIC.json).
- [Embeddings numériques, sans objets Python](NOISE_DIAGNOSTIC_FEATURES.npz).
- `reports/latest_noise_sensitivity_diagnostic.json` vérifie le rapport.
- 150 tests réussis, dont six sur les statistiques diagnostiques ; Ruff réussi.

Les originaux, résultats T1/T2, caches et paramètres du modèle sont conservés.
