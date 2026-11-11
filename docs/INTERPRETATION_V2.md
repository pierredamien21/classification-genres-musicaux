# Interprétation de l'évaluation V2 (phase 6)

Mis à jour : 11 novembre 2026.

Il s'agit d'une analyse descriptive de l'évaluation unique du 11 novembre
([rapport](../reports/v2/final_evaluation/report.json)). Elle ne modifie aucun modèle, aucun
seuil et aucune décision. Les chiffres et les méthodes se trouvent dans
[`reports/v2/interpretation.json`](../reports/v2/interpretation.json), produit par
`scripts/interpret_v2.py`.

## 1. Une réplication, pas un gain sur le test V1

Les exports V1 n'ont pas changé. Ils sont mesurés sur un test neuf : 501 morceaux, un par artiste,
venant d'artistes absents de FMA-small.

| Système V1 figé | Validation V1 | Test V1 | Test V2 [IC 95 %] |
|---|---:|---:|---:|
| ViT bloc 6 + HPSS (candidat V1) | 0,598 | 0,504 | 0,481 [0,442 ; 0,525] |
| ViT bloc 12 original | 0,545 | 0,464 | 0,453 [0,418 ; 0,496] |
| ViT bloc 6 original | 0,578 | 0,488 | 0,477 [0,441 ; 0,523] |
| CNN, graines 42 / 43 / 44 | 0,546 / 0,547 / 0,529 | 0,454 / 0,442 / 0,400 | 0,456 / 0,418 / 0,421 |

Sur le test V2, chaque système retrouve à peu près son score du test V1 : la différence va de −0,02 à +0,02.

**H1** (candidat − ViT original) passe de +0,039 [+0,005 ; +0,072] en V1 à +0,028
[−0,007 ; +0,064] en V2. L'effet garde le même sens et une ampleur voisine, mais il n'est plus
significatif. La puissance était d'environ 56 % : c'est un résultat **non concluant, pas une
réfutation**.

Le gain de MAEST (H2 et H3) porte sur un système nouveau. Il ne dit rien de la V1.

## 2. Pourquoi la V1 perdait 9 points entre validation et test

La validation croisée V2 a produit des têtes **jamais choisies sur la validation V1**. Appliquées
aux mêmes morceaux, elles retrouvent les scores officiels :

| Système | Validation V1 | Test V1 | Écart | IC 95 % (bootstrap par artiste) |
|---|---:|---:|---:|---:|
| ViT bloc 6 + HPSS | 0,595 (officiel 0,598) | 0,501 (officiel 0,504) | 0,094 | [+0,001 ; +0,188] |
| ViT bloc 12 | 0,545 (officiel 0,545) | 0,462 (officiel 0,464) | 0,083 | [−0,012 ; +0,180] |
| MAEST bloc 12 | 0,669 | 0,651 | 0,018 | [−0,070 ; +0,103] |

L'écart de la V1 ne vient donc **pas d'un surapprentissage de la sélection** sur la
validation. Il vient de la composition des deux ensembles :
- avec environ 780 morceaux regroupés par artiste, l'aléa d'échantillonnage est très large (± 9 points) ;
- les traits du ViT sont plus sensibles à cette composition que ceux de MAEST.

## 3. Pourquoi tout baisse entre le développement et le test V2

MAEST passe de 0,684 en validation croisée à 0,572 sur le test, et le candidat V1 de 0,553 à
0,481. Trois pistes ont été mesurées.

**Pondération par artiste.** Le développement compte en moyenne 3,4 morceaux par artiste. Si
l'on donne à chaque artiste le même poids, MAEST tombe à 0,654 et le candidat V1 à 0,518. Cette
pondération explique donc entre un quart et la moitié de l'écart.

**Métadonnées.** Le test est moins écouté (médiane de 374 écoutes contre 2 493) et vient
d'artistes moins prolifiques. En revanche, son débit et son nombre de genres FMA sont les mêmes,
et ses années sont proches. Repondérer le développement selon ces variables laisse son exactitude
entre 0,67 et 0,70, contre 0,579 sur le test. Le test est plus difficile **dans chaque tranche** :
ces métadonnées n'expliquent pas le décalage.

**Genres.** Pour MAEST, trois genres changent nettement :
- Instrumental s'effondre (0,60 → 0,27) ;
- Electronic recule (0,74 → 0,60) ;
- Hip-Hop progresse (0,85 → 0,91).

Les confusions qui augmentent le plus vont vers **Experimental** : +26 points depuis
Instrumental, +24 depuis Electronic, +10 depuis Pop. Le candidat V1 montre le même schéma, ce qui
désigne la population du test plutôt qu'un modèle.

Une explication reste plausible sans être établie. Pour des artistes peu écoutés de FMA-large, la
frontière entre Instrumental, Electronic et Experimental serait plus floue, que ce soit par le
style ou par l'annotation. Seule l'écoute peut éclairer ce point.

## 4. Écoute annotée (faite par l'utilisateur le 11 novembre)

Les 18 extraits ont été écoutés et notés **avant** l'affichage des étiquettes. Les observations
sont dans [`reports/v2/listening/observations.json`](../reports/v2/listening/observations.json)
et la sélection dans [`queue.json`](../reports/v2/listening/queue.json).

Il y a un seul auditeur, non spécialiste, et l'échantillon est ciblé. Ce qui suit est donc
qualitatif : aucune étiquette n'est modifiée et aucune fréquence n'est estimée.

- **Contrôles cohérents (3 sur 3).** Les deux morceaux Hip-Hop sont décrits comme « du rap ».
  Le morceau Folk est décrit comme une « guitare qui joue une mélodie calme ».
- **« Experimental » attire les morceaux sans repères nets.** Quatre erreurs envoyées vers
  Experimental, souvent avec une forte confiance, sont décrites ainsi : « rien n'est
  distinguable », « bizarre », « aucune idée ». Il s'agit de Pop à 0,94, Instrumental à 0,61 et
  0,78, et Electronic à 0,91. Le modèle et l'auditeur y perçoivent la même absence de marqueurs
  de genre. C'est cohérent avec la bascule observée sur le test V2.
- **Des erreurs franches du modèle.** Trois morceaux Instrumental sans voix sont prédits
  Experimental. L'auditeur entend « guitare, tam-tam », « violon ou piano », « ambiance de bar ».
  L'étiquette FMA y paraît plausible.
- **Des étiquettes discutables.**
  - Un morceau « Instrumental » contient une voix parlée sur une guitare électrique.
  - Un morceau « Classical » (hors domaine) est décrit comme « une guitare électrique à fond ».
- **Un chevauchement de la taxonomie.** Un morceau International est une chanson calme en
  espagnol, prédite Folk à 0,87 par les deux candidats. Les deux étiquettes se défendent.
- **Un hors domaine évident, mais accepté.** Un morceau Classical joué au violon et au piano,
  sans batterie, passe le seuil de rejet avec une confiance de 0,47 (seuil 0,337). Le seuil
  réglé sur le développement est trop permissif pour un modèle qui ne connaît pas la musique
  classique.

**Lecture.** L'écoute ne contredit pas l'explication de la section 3. Le flou de la frontière
Instrumental / Electronic / Experimental mêle des erreurs du modèle, des morceaux peu typés et
quelques étiquettes discutables. Ces observations ne permettent pas de les compter.

La fiche se trouve dans `data/interim/error-review-v2-20261111T144209718257Z/index.html`.

Elle contient 18 extraits de 18 artistes, en six groupes de trois :
- erreurs dont le modèle est sûr ;
- erreurs Pop ;
- erreurs Instrumental ;
- faux Experimental ;
- contrôles correctement classés ;
- morceaux hors domaine acceptés.

Le tirage utilise la graine 20261111. Les audios sont intégrés à la page et leurs empreintes sont
vérifiées.

1. Ouvrir la page dans un navigateur.
2. Écouter chaque extrait **avant** d'afficher son étiquette.
3. Noter les voix, le rythme, les instruments et les genres plausibles.
4. Exporter le JSON dans le même dossier.

L'échantillon est ciblé : il ne mesure pas la fréquence des erreurs. Aucune étiquette ne sera
modifiée.

## Limites

- Toutes ces analyses sont descriptives, faites après le test et non préenregistrées.
- Les têtes de validation croisée ont appris sur 80 % du développement, contre 100 % pour les modèles V1.
- L'analyse des métadonnées ne porte que sur MAEST.
- L'écoute repose sur un seul auditeur non spécialiste et 18 extraits ciblés. Elle ne mesure aucune fréquence.
