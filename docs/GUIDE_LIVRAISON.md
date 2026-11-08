# Prendre en main le projet

7 novembre 2026. Campagne scientifique V1 et session utilisateur terminées et vérifiées.
Les notebooks, la démo locale et la [conclusion générale](CONCLUSION_PROJET.md)
sont disponibles. [Résultats de la session](../reports/SESSION_UTILISATEUR.md) :
macro-F1 validation 0,5707, meilleure époque 26 sur 30. Aucun nouveau lancement requis ;
les commandes ci-dessous restent la procédure de reproduction et de consultation.

## Comprendre et présenter

Ouvrir [le parcours HTML](../notebooks/00_parcours_projet.html), puis
[l’analyse audio](../notebooks/02_analyse_audio.html). Les sorties sont enregistrées.
Le [rapport final V1](../reports/EVALUATION_FINALE.md) et la [fiche modèle](FICHE_MODELE.md)
portent les résultats : macro-F1 0,5035, accuracy 51,16 % sur test. La baisse face
à la validation et les comparaisons secondaires incertaines restent visibles.

Le [scénario portfolio](PRESENTATION_PORTFOLIO.md) fournit une trame orale de trois
minutes. Une vidéo n’a pas été enregistrée ni publiée. La diffusion du corpus ou
des poids nécessite d’en conserver et vérifier les licences.

## Essayer un fichier

Depuis la racine du projet dans PowerShell :

```powershell
.\scripts\ouvrir_demo.ps1
```

Ouvrir <http://127.0.0.1:7860>, charger un WAV/MP3, choisir les passages, puis
cliquer **Analyser le morceau**. Limites : 50 Mo, cinq minutes, huit genres,
scores non calibrés. Le mode début/milieu/fin ne possède pas une estimation test
séparée ; le mode FMA conserve les positions 0/10/20 s. Ctrl+C ferme le serveur.
Si la politique PowerShell bloque le script, utiliser directement :

```powershell
.\.venv-demo\Scripts\python.exe app/main.py
```

Aucun lien public créé. Les copies téléversées restent dans
`data/interim/demo-uploads` ; purge toutes les dix minutes des fichiers âgés de
plus d’une heure pendant que le serveur fonctionne. Une fermeture anticipée
peut laisser des copies locales. Cette démo n’est pas préparée comme service public.

## Lancer ta session autonome

```powershell
.\scripts\ouvrir_notebooks.ps1
```

Ouvrir `notebooks/05_entrainement_utilisateur.ipynb`, noyau **Music genres — session
utilisateur**. Lire les vérifications, passer `LANCER=True`, exécuter cette cellule,
puis remettre `LANCER=False`. Le calcul continue hors du notebook.

Recette : CNN depuis zéro, graine 42, plafond 30, patience 5, amélioration strictement
positive, meilleure macro-F1 validation conservée. Environ 35–80 minutes au plafond
selon la charge, arrêt potentiellement plus tôt ; budget d’apprentissage de deux
heures, contrôles et export en supplément. Brancher le PC et éviter la veille.

Alternative sans notebook :

```powershell
.\.venv-notebook\Scripts\python.exe scripts/user_session.py --launch
.\.venv-notebook\Scripts\python.exe scripts/user_session.py
```

La deuxième commande consulte l’état sans relancer. Après une interruption,
inspecter le journal, puis reprendre explicitement :

```powershell
.\.venv-notebook\Scripts\python.exe scripts/user_session.py --launch --resume
```

Ne pas changer sources, données, versions ou paramètres entre lancement et reprise.
Conserver le dossier indiqué par `reports/latest_cnn_user_training.json`,
`reports/USER_TRAINING_JOB.json` et les journaux `data/interim/cnn_user-*.log`.
Les checkpoints à charger sont ceux de ce projet local.

Quand l’état devient `complete`, dire : **« La session utilisateur est terminée,
vérifie ses sorties et finalise le projet. »** En cas d’échec, conserver les traces
et signaler le problème. Les scores du petit essai de préparation ne rejoignent
pas le classement scientifique.

## Environnements et reproduction

| Dossier | Usage | Vérification |
|---|---|---|
| `.venv` | V1, notebooks 00/02, tests | Versions et sources V1 inchangées |
| `.venv-notebook` | JupyterLab, notebook 05, CNN utilisateur | Copies physiques indépendantes des paquets ; imports et pause/reprise testés |
| `.venv-demo` | Gradio 6.31 | Pillow 12.3 spécifique à la démo ; prédiction comparée à V1 avec tolérance 10⁻⁶ |

Les nouveaux environnements ont été créés avec un interpréteur Python puis des
copies physiques des paquets locaux ; leurs `site-packages` sont indépendants.
Ils partagent les sources et les données du projet. Ce contrôle n’est pas une
installation neuve depuis Internet sur une autre machine.

Versions exactes : `requirements/notebook-windows-py313.txt` et
`requirements/demo-windows-py313.txt`. Exemple de reconstruction future avec Python 3.13 :

```powershell
uv venv --python 3.13 .venv-reproduction
uv pip install --python .venv-reproduction/Scripts/python.exe -r requirements/notebook-windows-py313.txt
```

Cette reconstruction réseau reste non testée ; les mêmes index et l’architecture
x86-64 peuvent être nécessaires. Ne pas remplacer un environnement vérifié sans
les contrôles d’identité. Les caches et poids volumineux sont exclus de Git :
suivre [les données](REPRODUCTION_REVUE.md), [l’environnement](ENVIRONNEMENT.md),
les protocoles d’extraction et le manifeste verrouillé.

## Dernière étape réalisée

Session complète contrôlée, courbes et résultats intégrés, conclusion générale rédigée.
La recette utilisateur reste un développement postérieur à la consultation du test V1,
sans nouvelle évaluation indépendante sur ce même test.
Publication publique, vidéo enregistrée et écoute subjective annotée restent
explicitement non réalisées.
