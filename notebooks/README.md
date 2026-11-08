# Notebooks du projet

**Actualisation du 7 novembre :** résultats complets intégrés à 00 et 05, courbes et conclusion ajoutées. Le notebook 05 consulte la session terminée avec `LANCER=False`. Le paragraphe du 5 novembre décrit la première livraison.

Mis à jour le 5 novembre 2026. Trois supports existent et ont été exécutés dans
l’ordre avec un noyau neuf. Le troisième conserve son lancement désactivé.

| Support | Usage | État |
|---|---|---|
| [00 — Parcours](00_parcours_projet.ipynb) · [HTML](00_parcours_projet.html) | Question, littérature, données, validation, test et limites | Exécuté, sorties enregistrées |
| [02 — Analyse audio](02_analyse_audio.ipynb) · [HTML](02_analyse_audio.html) | Même passage train : signal, STFT, Mel, MFCC, chroma, rythme, HPSS, tenseur réel du ViT | Exécuté ; dix figures au total avec 00 |
| [05 — Session utilisateur](05_entrainement_utilisateur.ipynb) | Lancement volontaire, suivi et lecture des sorties | Session complète vérifiée ; sorties actualisées avec LANCER=False |

Les approfondissements 01/03/04 du plan initial sont intégrés au parcours et
aux rapports liés ; ils ne sont pas nécessaires comme fichiers séparés.

## Ouvrir JupyterLab

Depuis la racine dans PowerShell :

```powershell
.\scripts\ouvrir_notebooks.ps1
```

Commande directe si la politique PowerShell empêche les scripts :

```powershell
.\.venv-notebook\Scripts\python.exe -m jupyterlab --ip=127.0.0.1
```

Choisir **Music genres — V1** pour 00/02 et **Music genres — session utilisateur**
pour 05. Les deux noyaux sont enregistrés dans `.venv-notebook/share/jupyter/kernels`.
Les notebooks trouvent la racine dans les parents du dossier courant ; pour
un autre emplacement, définir `MUSIC_GENRE_ROOT` vers la racine du projet.

## Rejouer les supports

00 lit les rapports épinglés. 02 décode un morceau train, calcule les illustrations
et un embedding gelé. Aucun entraînement, téléchargement ni décodage test dans
00/02. Aucun audio intégré aux sorties. 05 conserve `LANCER=False` par défaut.

```powershell
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks/00_parcours_projet.ipynb notebooks/02_analyse_audio.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to html notebooks/00_parcours_projet.ipynb notebooks/02_analyse_audio.ipynb
.\.venv-notebook\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks/05_entrainement_utilisateur.ipynb
```

Ne pas réexécuter la dernière commande en laissant `LANCER=True`. Jupyter nécessite
des connexions locales, à autoriser dans le bac à sable de l’assistant, sans
serveur public. Les exécutables globaux `jupyter-*` du poste ont montré une
incompatibilité ; cibler directement les modules des environnements ci-dessus.

Les exports HTML sont créés et leur contenu contrôlé sur disque. Leur ouverture
via `file://` a été refusée par le navigateur intégré de l’assistant : leur rendu
dans ce navigateur n’est pas annoncé comme testé. Les figures sont également
exportées en PNG/SVG dans `reports/notebooks-v1/figures`.

[Guide de livraison](../docs/GUIDE_LIVRAISON.md) · [Conception](../docs/CONCEPTION_NOTEBOOKS.md).
