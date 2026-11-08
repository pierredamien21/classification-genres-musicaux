# Incertitude par artiste — contrôle sur validation

Mis à jour : 28 octobre 2026. T3 implémentée et vérifiée sur les prédictions de validation sauvegardées. Aucun audio chargé, aucun entraînement ni résultat test consulté.

## Ce que montrent ces résultats

Le candidat bloc 6 HPSS gagne 0,0528 de macro-F1 sur le ViT bloc 12 original ; l’intervalle apparié obtenu est [0,0257 ; 0,0785]. Les écarts avec le bloc 12 HPSS et le bloc 6 original sont plus petits et leurs intervalles de macro-F1 contiennent zéro. La fusion reste moins bonne en score ponctuel que le ViT original, mais son intervalle contient également zéro.

Ces résultats décrivent la variabilité selon les artistes échantillonnés, conditionnellement aux modèles déjà fixés. Ils ne démontrent pas une supériorité sur des données indépendantes : cette validation a servi à sélectionner les variantes, C et les époques. Aucun modèle n’est remplacé et aucun réglage n’est choisi à partir des intervalles.

## Méthode fixée avant le calcul

- 780 morceaux de validation, regroupés en 219 artistes ; un à 55 morceaux par artiste, dont 111 artistes avec un seul morceau.
- 2 000 tirages PCG64, graine 20261028 : 219 artistes avec remise à chaque tirage. Ordre canonique lexicographique des identifiants d’artiste et de morceau.
- Tous les morceaux de chaque artiste sélectionné sont conservés, avec multiplicité. Mêmes tirages pour tous les modèles ; ni tirage de fenêtres, ni tirage interne aux artistes, ni stratification par genre.
- Macro-F1 sur huit labels fixes, divisions indéfinies à zéro ; accuracy par morceau. Les artistes sont l’unité de tirage, pas l’unité de moyenne des scores.
- Intervalles percentiles ponctuels à 95 %, quantiles 0,025/0,975 et interpolation linéaire. Différences calculées dans chaque tirage avant les quantiles.
- CNN : trois métriques de graines sur le même tirage, puis moyenne ; aucune moyenne des probabilités et aucun rééchantillonnage des graines.

Les tirages contiennent de 499 à 1085 morceaux : cet effectif variable est attendu avec des groupes inégaux. Aucun des 2 000 tirages n’a de classe absente ; le code les aurait conservés avec les huit labels, ce que couvrent les tests.

## Scores et intervalles

| Système | Macro-F1 | Intervalle 95 % | Accuracy | Intervalle 95 % |
|---|---:|---:|---:|---:|
| Descripteurs | 0.4612 | [0.3871 ; 0.5042] | 0.4756 | [0.4035 ; 0.5476] |
| ViT bloc 12 original | 0.5447 | [0.4542 ; 0.5810] | 0.5538 | [0.4668 ; 0.6289] |
| Fusion bloc 12 + descripteurs | 0.5219 | [0.4353 ; 0.5577] | 0.5295 | [0.4530 ; 0.5980] |
| ViT bloc 6 original | 0.5780 | [0.4822 ; 0.6161] | 0.5885 | [0.5043 ; 0.6598] |
| ViT bloc 12 HPSS | 0.5844 | [0.4759 ; 0.6283] | 0.5974 | [0.5065 ; 0.6752] |
| ViT bloc 6 HPSS | 0.5975 | [0.5034 ; 0.6352] | 0.6128 | [0.5300 ; 0.6847] |
| CNN vingt, graine 42 | 0.5464 | [0.4450 ; 0.5855] | 0.5615 | [0.4676 ; 0.6361] |
| CNN vingt, graine 43 | 0.5471 | [0.4525 ; 0.5897] | 0.5590 | [0.4719 ; 0.6355] |
| CNN vingt, graine 44 | 0.5287 | [0.4410 ; 0.5666] | 0.5474 | [0.4594 ; 0.6246] |
| Majoritaire de train | 0.0284 | [0.0159 ; 0.0442] | 0.1282 | [0.0681 ; 0.2148] |
| CNN vingt, moyenne des métriques | 0.5407 | [0.4492 ; 0.5782] | 0.5560 | [0.4689 ; 0.6289] |

La dispersion entre les trois graines CNN est distincte de l’incertitude par artiste : macro-F1 moyenne 0,5407, écart-type d’échantillon 0,0105. L’intervalle de la moyenne ci-dessus conditionne sur ces trois réseaux précis ; il ne couvre pas la variabilité de tous les entraînements possibles.

## Différences appariées annoncées

| Comparaison (gauche − droite) | Écart macro-F1 | Intervalle 95 % | Écart accuracy | Intervalle 95 % |
|---|---:|---:|---:|---:|
| ViT bloc 6 HPSS − ViT bloc 12 original | +0.0528 | [+0.0257 ; +0.0785] | +0.0590 | [+0.0337 ; +0.0902] |
| Fusion bloc 12 + descripteurs − ViT bloc 12 original | -0.0228 | [-0.0490 ; +0.0073] | -0.0244 | [-0.0551 ; +0.0097] |
| ViT bloc 6 HPSS − ViT bloc 12 HPSS | +0.0131 | [-0.0144 ; +0.0416] | +0.0154 | [-0.0113 ; +0.0428] |
| ViT bloc 6 HPSS − ViT bloc 6 original | +0.0195 | [-0.0098 ; +0.0436] | +0.0244 | [+0.0012 ; +0.0497] |

La première comparaison est principale ; les trois autres sont secondaires. Les intervalles sont ponctuels, sans correction simultanée. Un intervalle contenant zéro ne prouve pas une équivalence. Celui de l’accuracy candidat − bloc 6 original reste positif, contrairement à celui de macro-F1 : conserver les deux résultats et le critère principal annoncé. Ne pas soustraire les bornes des intervalles individuels pour obtenir un intervalle de différence.

## Vérifications et reproduction

178 tests réussis, dont 23 pour le bootstrap : groupes inégaux, multiplicité, classes absentes, alignement, permutations, graine, différence avec soi-même nulle et moyenne de métriques. Exports, probabilités sauvegardées et manifeste vérifiés par empreintes ; scores ponctuels reproduits pour les neuf exports.

Le premier essai a identifié une tolérance de somme trop stricte dans le nouveau module (10⁻⁸) pour les exports CNN : écart maximal observé 1,36 × 10⁻⁷. Le contrôle utilise désormais la tolérance 10⁻⁶ du protocole existant, sans renormalisation ni modification des scores ; un test de régression accepte l’arrondi et rejette un écart de 2 × 10⁻⁵.

Le vérificateur indépendant régénère les tirages, développe explicitement les listes de morceaux avec répétitions, recalcule les scores avec scikit-learn et interpole les quantiles à partir des valeurs triées. Il vérifie les 2 000 tirages, les 44 000 valeurs de métriques et les intervalles des quatre différences.

```powershell
.\.venv\Scripts\python.exe scripts/check_bootstrap_validation.py
.\.venv\Scripts\python.exe scripts/verify_bootstrap_validation.py
```

[Rapport machine et provenance](BOOTSTRAP_VALIDATION.json) · [Tirages et distributions](BOOTSTRAP_VALIDATION_DRAWS.npz) · [Vérification indépendante](BOOTSTRAP_VALIDATION_CHECK.json)

## Décision et suite

T3 est satisfaite. Conserver le candidat et les comparaisons annoncées ; passer à T4, vérification des autres adaptateurs linéaires (descripteurs, ViT bloc 12, fusion et HPSS bloc 12), puis T5 pour les CNN. Le bloc 6 original est déjà adapté par T2b ; la vérification groupée des comparateurs reste à terminer. Le protocole demeure non verrouillé et le test fermé.
