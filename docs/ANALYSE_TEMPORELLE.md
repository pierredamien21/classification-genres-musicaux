# Analyse temporelle : comprendre les désaccords entre passages

Mis à jour : 27 octobre 2026. État : calcul terminé sur les 780 morceaux de
validation (219 artistes), avec trois fenêtres par morceau et trois ViT fixes.
Documentation finalisée après l’interruption du service de vérification automatique.

## Question et protocole

Le diagnostic ciblé trouvait une fenêtre correcte dans sept des quatorze erreurs
HPSS sélectionnées. Cette proportion pouvait être biaisée par la sélection des
cas. La nouvelle analyse porte donc sur toute la validation, sans nouvelle
extraction, entraînement ou optimisation. Les probabilités sont recalculées à
partir des embeddings vérifiés et des classificateurs exportés.

La [configuration](../configs/temporal_analysis.json), écrite avant exécution,
fixe la cohorte, les trois modèles, les positions 0/10/20 s et une tolérance
absolue de 0,00001 pour retrouver les probabilités historiques. Les modèles sont
toujours ceux entraînés sur trois fenêtres ; les résultats par position ne sont
pas des modèles entraînés séparément sur chaque passage.

Comptages : désaccords entre les trois genres prédits, nombre de fenêtres
correctes croisé avec la réussite de la moyenne, différence entre majorité
unique et moyenne des probabilités, détail par genre. Trois votes différents
n’ont pas de majorité unique. Une égalité d’argmax suit l’ordre des classes.
La vérité terrain intervient uniquement dans l’évaluation, jamais pour choisir
une fenêtre ou produire une prédiction.

## Résultats utiles à la décision

| Modèle | Désaccords / 780 | Erreurs agrégées | Parmi ces erreurs, ≥1 fenêtre correcte |
|---|---:|---:|---:|
| Bloc 12 original | 429 | 348 | 105 |
| Bloc 6 original | 346 | 321 | 80 |
| Bloc 12 HPSS | 392 | 314 | 73 |

Pour HPSS, 73/314 erreurs (23,2 %) comportent une fenêtre correcte, contre 7/14
dans la sélection ciblée. Dans 241 erreurs, aucun des trois argmax n’est correct.
Inversement, la moyenne récupère un morceau sans fenêtre individuellement
correcte. Ces résultats montrent pourquoi un simple comptage des erreurs
« récupérables » ne prouve pas qu’un sélecteur saura les corriger.

La moyenne des trois probabilités dépasse chaque position seule en macro-F1
et en accuracy pour chacun des trois modèles. Avec HPSS : **0,5844** en macro-F1
pour la moyenne, contre **0,5323 / 0,5129 / 0,5330** par position. Cela justifie
de conserver la moyenne comme référence ; aucune supériorité sur toutes les
autres méthodes d’agrégation n’est démontrée.

Pour la **Pop**, HPSS produit 82 erreurs sur 94 morceaux. Dix erreurs contiennent
une fenêtre correcte et 72 n’en contiennent aucune. Pour **Hip-Hop**, ces nombres
sont respectivement 29 erreurs sur 100, dont neuf avec une fenêtre correcte.
Une modification de l’agrégation ne peut donc pas être supposée résoudre à elle
seule la faiblesse sur la Pop. Les représentations, l’apprentissage et les
étiquettes restent des hypothèses à examiner, sans cause établie ici.

## Décision et suite

Conserver les trois fenêtres et leur moyenne. Ne pas lancer une recherche de
poids temporels à partir de ces erreurs. La prochaine étape est de préparer le
protocole et le benchmark de **bloc 6 + HPSS**, piste déjà inscrite au registre :
comparer à bloc 6 original et bloc 12 HPSS, même cohorte, grille et agrégation.
Cette combinaison reste non testée et son gain n’est pas présumé. Le benchmark
sur 64 fenêtres de train doit précéder une nouvelle extraction complète.

## Vérifications, preuves et limites

Les empreintes des caches, modèles, prédictions et sources d’extraction sont
vérifiées ; les versions d’extraction sont contrôlées. Identifiants, labels,
artistes, sources audio et minutages concordent entre manifeste et caches.
Les moyennes reproduisent les métriques historiques exactement, et leurs
probabilités à **2,22 × 10⁻¹⁶** près. Aucun fichier audio n’a été redécodé et
aucune caractéristique du test n’a été chargée pour cette analyse.

La suite complète compte **108 tests réussis** ; Ruff passe. Neuf cas nouveaux
couvrent un exemple chiffré avec réordonnancement, la différence moyenne/vote,
le refus du train/test, des fenêtres ou métadonnées incohérentes et des
probabilités invalides. La figure PNG a été inspectée ; le PDF a été généré.

Les statistiques restent descriptives sur un jeu de développement déjà utilisé
pour sélectionner les modèles. Les artistes ne sont pas indépendants morceau
par morceau ; aucun intervalle ni résultat significatif n’est annoncé. Les
désaccords ne constituent ni une écoute ni une preuve de changement de genre.

- [Tableaux complets et figure](../reports/ANALYSE_TEMPORELLE.md).
- [Mesures et probabilités par morceau](../reports/TEMPORAL_ANALYSIS.json).
- [Pointeur avec empreinte](../reports/latest_temporal_analysis.json).
- [Code des comptages](../src/music_genre/evaluation/temporal.py).

Reproduction depuis la racine :

```powershell
.\.venv\Scripts\python.exe scripts/analyze_temporal_validation.py
```

La commande crée un dossier daté contenant PNG, PDF et CSV de comptages, puis
actualise le rapport numérique, son pointeur et le rapport Markdown.
