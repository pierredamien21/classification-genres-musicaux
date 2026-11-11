# Rapport V2 : genres musicaux, réplication et modèles audio préentraînés

Mis à jour : 11 novembre 2026. Ce rapport synthétise les phases 0 à 6 du [plan V2](PLAN_V2.md). Les chiffres viennent des rapports cités, qui portent les empreintes.

## Question

La V1 concluait qu'un ViT d'images gelé, appliqué à des spectrogrammes (bloc 6 + HPSS), battait le ViT d'origine sur FMA-small : +0,039 de macro-F1 sur 778 morceaux de test. La [critique](archive/CRITIQUE_PROJET.md) relevait trois faiblesses :
- un test concentré sur peu d'artistes ;
- l'absence de comparateur audio préentraîné ;
- un CNN de référence faible.

La V2 pose trois hypothèses, préenregistrées avant tout téléchargement ([protocole](PROTOCOLE_V2.md)) :
- **H1 (primaire)** : l'effet V1 se réplique-t-il sur un test neuf ?
- **H2** : un encodeur audio préentraîné gelé fait-il mieux que le candidat V1 ?
- **H3** : le système V2 retenu fait-il mieux qu'un CNN renforcé ?

## Protocole

- **Données** ([fiche](FICHE_DONNEES.md)) :
  - développement : 7 857 morceaux de FMA-small, découpés en 5 plis groupés par artiste ;
  - test V2 : 501 morceaux d'artistes absents de FMA-small, un par artiste ;
  - hors domaine : 340 morceaux, plus 12 morceaux Gospel.
- **Trois verrous par empreintes** :
  1. le protocole et les listes, avant le téléchargement ;
  2. l'audio audité et les cohortes, avant tout modèle ;
  3. les systèmes, les poids, le code et l'environnement, avant l'unique inférence ([verrou 3](../reports/v2/lock3.json)).
- **Datation** : chaque protocole est commité avant les calculs qu'il encadre. Exemple : la phase 4 est fixée par le commit `0cc8b5f` (10 novembre, 19 h 37). `phase4.json`, figé par le verrou 3, cite ce commit sous son identifiant antérieur à une réécriture de l'historique (`2cff310`) ; le contenu est le même.
- **Sélection** : validation croisée imbriquée sur le développement uniquement. La meilleure macro-F1 moyenne hors pli l'emporte ; en dessous de 0,002 d'écart, le système le moins coûteux gagne.
- **Statistiques** :
  - BCa sur 5 000 tirages ;
  - H1 au seuil α = 0,05 ;
  - Holm sur H2 et H3 ;
  - recomptage indépendant (scikit-learn, SciPy, ECE réécrite).

## Choix du système (développement uniquement)

| Système (sonde logistique sauf mention) | Macro-F1 en CV |
|---|---:|
| **MAEST bloc 12** | **0,684 ± 0,027** |
| MAEST bloc 12, tête MLP | 0,668 |
| AST bloc 12 | 0,654 |
| PANNs CNN14 | 0,627 |
| ViT natif bloc 6 (sans redimensionnement) | 0,569 |
| ViT bloc 6 + HPSS (candidat V1) | 0,553 |
| ViT bloc 12 original | 0,520 |
| Descripteurs musicaux | 0,469 |

Les encodeurs audio préentraînés dominent nettement. Le ViT d'images, affiné sur ses blocs 5 et 6, n'atteint que 0,661, et le CNN renforcé 0,618 : ces deux scores viennent du pli 0 seul, qui est un pli facile. Preuves : [cv_results.csv](../reports/v2/cv_results.csv), [candidate.json](../reports/v2/candidate.json).

## Robustesse et calibration (développement, phase 4)

- **Robustesse de MAEST** : il résiste au gain (±6 dB, −0,006) et au MP3 à 64 kb/s (−0,004). Il cède au bruit blanc fort : −0,19 à 10 dB.
- **Augmentation** : elle comblait presque cet écart (+0,16 à 10 dB). Elle n'a pas été adoptée, car la règle fixée d'avance portait sur l'audio propre (0,681 < 0,6855). L'auteur a maintenu la règle.
- **Calibration** : température 0,824. Seuil de rejet 0,337, qui écarte 5 % du développement. Preuves : [phase4.json](../reports/v2/phase4.json).

## Résultats sur le test V2 (inférence unique)

| Hypothèse | Différence de macro-F1 | IC 95 % | Verdict |
|---|---:|---:|---|
| H1 : ViT bloc 6 + HPSS − ViT original | +0,028 | [−0,007 ; +0,064] | Non concluante (p = 0,12) |
| H2 : MAEST − ViT bloc 6 + HPSS | +0,091 | [+0,049 ; +0,134] | Confirmée (p Holm = 0,0003) |
| H3 : MAEST − CNN renforcé (3 graines) | +0,117 | [+0,083 ; +0,154] | Confirmée (p Holm = 0,0003) |

- **Scores sur le test** :
  - MAEST : 0,572 ;
  - candidat V1 : 0,481 ;
  - CNN V2 : 0,444–0,462 ;
  - CNN V1 : 0,418–0,456.
- **Sensibilité** : sans les 19 morceaux signalés par le décodeur, H1 vaut +0,024. Aucune conclusion ne change.
- **Résultats défavorables** :
  - la température dégrade l'ECE sur le test (0,045 → 0,085) ;
  - le seuil accepte 89 % du hors domaine (AUROC 0,67) et 11 morceaux Gospel sur 12.
- Preuves : [rapport](../reports/v2/final_evaluation/report.json), [recomptage](../reports/v2/final_evaluation/recount.json), [prédictions](../reports/v2/final_evaluation/predictions.csv).

## Interprétation ([détail](INTERPRETATION_V2.md))

- **Réplication** : les modèles V1 retrouvent sur le test V2 leur score du test V1, à 1–2 points près. H1 garde son sens et une ampleur voisine, mais sa puissance (environ 56 %) ne permettait pas de trancher.
- **Écart validation/test de la V1 (9 points)** : des têtes jamais choisies sur la validation V1 le reproduisent. Il vient de la composition par artiste, pas d'un surapprentissage de la sélection.
- **Baisse entre développement et test V2** :
  - 3 à 4 points viennent de la pondération par artiste ;
  - les métadonnées n'en expliquent presque rien ;
  - Instrumental et Electronic basculent vers Experimental, chez tous les modèles.
- **Écoute de 18 extraits par l'auteur** : elle mêle trois cas :
  - des morceaux peu typés, que l'auditeur décrit lui aussi sans repères ;
  - des erreurs franches ;
  - quelques étiquettes discutables.

## Conclusions

1. Le gain de la V1 n'est **ni confirmé ni réfuté**. Son ampleur sur un test neuf reste voisine, mais n'est plus significative.
2. Un encodeur audio préentraîné gelé (MAEST) **dépasse nettement** le ViT d'images et le CNN entraîné de zéro, sur un test indépendant.
3. Les scores du développement **surestiment** la performance sur de nouveaux artistes d'environ 11 points, même avec une validation croisée groupée par artiste.
4. La calibration et le rejet réglés sur le développement **ne se transfèrent pas** au test.

## Limites

- **Tests** : un seul test V2 ; la puissance de H1 est modeste ; International ne compte que 53 morceaux de test.
- **CNN et ViT affiné** : évalués sur le pli 0 seul pendant le choix. Le CNN final a appris sur 80 % du développement.
- **MAEST** : licence non commerciale ; recouvrement invérifiable entre Discogs20 et FMA.
- **Robustesse** : mesurée sur des dégradations synthétiques (bruit blanc, gain, MP3 mono).
- **Analyses de la phase 6** : descriptives et non préenregistrées.

Le tableau qui relie chaque critique à son traitement est dans [CONTROLE_FINAL.md](CONTROLE_FINAL.md).
