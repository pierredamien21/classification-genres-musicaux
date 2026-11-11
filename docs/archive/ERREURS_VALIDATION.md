# Analyse des erreurs et revue d’écoute

Mis à jour : 26 octobre 2026. L’analyse quantitative est réalisée ; les observations
d’écoute restent à recueillir. Voir le [rapport](../../reports/ERREURS_VALIDATION.md).

## Pourquoi cette étape

Le meilleur score global ne décrit pas tous les genres : HPSS améliore la macro-F1,
mais dégrade certains morceaux Hip-Hop. Pop reste difficile pour les sept réseaux
ou classificateurs examinés. Avant d’attribuer cela au signal, à la représentation
ou aux annotations, nous séparons les faits mesurés des hypothèses.

Les trois ViT, les descripteurs et les trois CNN prolongés sont comparés sur les
mêmes 780 morceaux de validation. Rapports, modèles, prédictions et manifeste sont
contrôlés par empreintes ; les métriques sont recalculées. L’alignement utilise
l’identifiant du morceau et vérifie le genre réel. Les trois CNN restent séparés :
aucun ensemble ou choix de graine n’est ajouté.

## Résultats qui motivent la revue

- Pop : 71 des 94 morceaux sont manqués par les trois ViT. Avec HPSS, 12 morceaux
  sont correctement reconnus, 82 sont manqués et 41 autres morceaux sont prédits
  Pop à tort. Le problème concerne à la fois le rappel et la précision.
- Hip-Hop : HPSS corrige 3 erreurs du ViT bloc 12 original et dégrade 8 morceaux
  auparavant corrects. Le gain global ne s’applique donc pas uniformément.
- Sur l’ensemble de la validation, HPSS corrige 77 erreurs et en introduit 43.
  Ces différences ne prouvent pas une cause acoustique ni une erreur de label.

## Sélection des écoutes

`configs/error_review.json` fixe sept catégories et trois morceaux visés par
catégorie. La sélection parcourt d’abord les catégories rares, puis utilise une
empreinte déterministe avec graine fixe. Un artiste apparaît au plus une fois
dans tout l’échantillon. Cette contrainte laisse deux catégories à deux morceaux :
**19 extraits de 19 artistes**, sur un quota de 21.

Les exemples correctement reconnus servent de points de comparaison. Les
faux positifs Pop permettent d’examiner aussi les autres genres. L’échantillon
est ciblé ; ses proportions ne mesurent pas la prévalence des erreurs.

## Utiliser les fiches

Le dossier courant est référencé par `review_folder` dans
`reports/VALIDATION_ERRORS.json`, avec un lien direct dans le rapport Markdown.
Il contient `index.html`, `queue.json` et `annotations_template.json`.

1. Ouvrir manuellement `index.html` dans un navigateur local.
2. Écouter avant d’ouvrir les étiquettes ; noter voix, rythme, instruments,
   évolution temporelle, genres plausibles et incertitudes.
3. Afficher les métadonnées et les prédictions, puis compléter les observations.
4. Cocher uniquement les écoutes effectivement réalisées et exporter le JSON.
5. Conserver cet export avec le dossier de revue ; il porte son identifiant.

Les fichiers audio intégrés sont identiques aux originaux et leurs licences
sont affichées. Les annotations sont prévues pour être mémorisées localement
dans le navigateur ; l’export JSON permet de les conserver séparément.
Le générateur crée un nouveau dossier à chaque exécution et ne remplace pas
les fiches précédentes. Aucun import ni bilan automatique des annotations
humaines n’est encore implémenté.

## Vérifications et limites

Nouveau test demandé par l’utilisateur et tenté le **24 octobre 2026** :
`cua.createBrowserTab("iab", "file:///E:/projet/vision%20transformer/data/interim/error-review-20261021T233320701672Z/index.html", { visible: true })`.
Résultat : `The requested URL protocol is not allowed. Allowed protocols: "http:", "https:".`
La navigation est refusée avant tout accès au lecteur. Ce résultat ne constitue
pas un test du décodage MP3, de la sortie sonore ou du JavaScript de la page.
Le serveur Chrome DevTools MCP requis par le skill dédié n’est pas disponible
dans les outils exposés de cette session. Aucun contournement du refus n’a été tenté.

Ordre retenu pour le diagnostic ciblé :

1. Vérifier manuellement l’ouverture de la fiche, puis lecture, pause, déplacement
   dans l’extrait et son audible. Distinguer chargement refusé, erreur du lecteur
   et lecture qui avance sans son. Noter le navigateur et tout message d’erreur.
2. Vérifier que les étiquettes restent masquées avant leur ouverture volontaire ;
   essayer une note clairement marquée comme test et son export sans la présenter
   comme une annotation scientifique.
3. Examiner des erreurs et des exemples corrects parmi les fiches existantes :
   audio original, trois fenêtres et spectrogrammes correspondants.
4. Énoncer une hypothèse fondée sur ces observations, puis comparer une seule
   modification à la référence. Ne pas relancer tous les entraînements à l’identique.

Confirmation utilisateur consignée le 26 octobre 2026 : « ca marche », en réponse à la vérification de l’affichage et du son du premier extrait. Ces deux points sont validés manuellement par l’utilisateur. Pause, déplacement, persistance, export et autres extraits ne sont pas confirmés ; aucune annotation musicale n’a été reçue.

La structure HTML, les 19 contenus audio intégrés, la diversité des artistes,
les champs initialement vides, les étiquettes masquées par défaut et la syntaxe
JavaScript ont été contrôlés. Preuves :
[`ERROR_REVIEW_CHECKS.json`](../../reports/ERROR_REVIEW_CHECKS.json).

L’ouverture automatisée a été refusée par la politique du navigateur intégré,
qui n’autorise pas le protocole `file://`. Ce refus de l’automatisation reste distinct
de la confirmation manuelle de l’affichage et du premier son. Persistance et export
ne sont pas vérifiés. Aucune revue subjective des 19 extraits n’est documentée.

Ne pas modifier les étiquettes sur la seule base d’un désaccord avec un modèle.
Les probabilités ne sont pas calibrées. Le test n’est ni chargé pour cette analyse
ni évalué. La combinaison bloc 6 + HPSS reste une expérience possible, non réalisée ;
son protocole et son coût devront être établis avant toute extraction.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/analyze_validation_errors.py
```

Cette commande actualise le rapport et crée un nouveau dossier de revue. Pour
reprendre les écoutes existantes, conserver et réutiliser leur dossier au lieu
de régénérer les fiches.
