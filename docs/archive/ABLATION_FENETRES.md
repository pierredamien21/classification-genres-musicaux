# Une fenêtre centrale ou trois fenêtres pour le ViT

## Question et périmètre

L’ablation prévue au plan compare le ViT gelé entraîné et évalué sur trois fenêtres
de dix secondes à la même méthode utilisant uniquement [10,20] secondes.
La référence conserve [0,10], [10,20] et [20,30] secondes. Il s’agit de la fenêtre
centrale du découpage fixe FMA de trente secondes, pas d’un recalcul du milieu de
chaque fichier à partir de sa durée exacte.

Les deux systèmes utilisent exactement les mêmes **6 299 morceaux de train** et
**780 morceaux de validation**, le même checkpoint, les mêmes 192 caractéristiques,
le même prétraitement et la grille C = {0,1 ; 1 ; 10}. La version centrale compte
6 299 fenêtres de train et 780 de validation, contre 18 897 et 2 340 pour la référence.

L’ablation change simultanément les fenêtres disponibles pour l’apprentissage et
pour la moyenne des probabilités à l’évaluation. Elle ne mesure pas à elle seule
l’effet de cette moyenne indépendamment des données d’apprentissage.

## Construction et contrôles

`scripts/build_vit_central.py` dérive un cache du ViT existant. Le sélecteur exige
trois fenêtres aux positions 0, 10 et 20 secondes pour **chaque** morceau. Fenêtre
manquante, identifiants répétés, métadonnées incohérentes ou présence du test
déclenchent une erreur : aucun morceau n’est écarté silencieusement.

Les lignes correspondant à 10 secondes sont copiées sans changer les vecteurs.
Le numéro global de ligne est réindexé ; l’identifiant de fenêtre d’origine est
conservé. Le cache contient configuration du parent, empreinte de son rapport,
sélection, versions et empreintes du code. Une reconstruction identique conserve
le rapport et ses empreintes. Aucun audio n’est décodé et le ViT n’est pas réexécuté.

La variante `--features vit_central` du script d’entraînement vérifie cet index
contre le sous-ensemble central exact de la référence musicale. Elle ajuste un
nouveau scaler uniquement sur les fenêtres centrales de train, puis la même
régression logistique. L’export conserve `offsets_seconds: [10.0]` pour décrire
correctement le prétraitement requis à l’inférence.

## Résultats

| Apprentissage et évaluation | C retenu | Macro-F1 validation | Accuracy |
|---|---:|---:|---:|
| Trois fenêtres | 0,1 | **0,5447** | **0,5538** |
| Centrale seule | 0,1 | 0,4882 | 0,4962 |

L’écart observé est de **0,0566** en macro-F1 en faveur des trois fenêtres. La
référence réussit 86 morceaux manqués par la variante centrale ; celle-ci réussit
41 morceaux manqués par la référence. Toutes les valeurs de C sont conservées dans
[le rapport](../../reports/ABLATION_FENETRES.md).

Après le résultat principal, un contrôle complémentaire conserve le modèle
trois fenêtres, y compris son scaler et C, mais l’évalue sur la centrale seule.
Il obtient **0,4896** de macro-F1 et **0,4962** d’accuracy. La baisse existe donc
aussi avec des poids fixes. Ce diagnostic ne réentraîne rien, ne sélectionne pas
de nouveaux paramètres et n’est pas adopté comme candidat au test. Il est marqué
explicitement comme complémentaire et postérieur au résultat principal.

Les trois fenêtres restent le choix de référence. Une seule fenêtre demande un
passage encodeur au lieu de trois, mais aucune accélération totale de facteur
trois n’est revendiquée : le décodage et le chargement du modèle ont leur propre coût.

## Reproduction

Depuis la racine, avec le cache ViT déjà calculé :

```powershell
.\.venv\Scripts\python.exe scripts/build_vit_central.py
.\.venv\Scripts\python.exe scripts/train_descriptor_baseline.py --features vit_central
.\.venv\Scripts\python.exe scripts/compare_vit_windows.py
```

Configuration du classifieur : `configs/baseline_vit_central.json`. Pointeurs :
`data/features/latest_vit_central.json` et `reports/latest_vit_central_run.json`.
La comparaison vérifie les empreintes, l’égalité des vecteurs avec les lignes du
parent, les morceaux et classes, la grille et les prédictions. Le modèle exporté
est rechargé et comparé aux probabilités scikit-learn sur toute la validation.

La validation sert au développement ; aucun écart significatif ni résultat final
indépendant n’est affirmé. Le test reste non évalué. Prochaine expérience prévue :
bloc 6 contre bloc 12 du ViT, à trois fenêtres et grille identiques, après benchmark.
