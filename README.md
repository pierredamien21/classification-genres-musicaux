# Genres musicaux avec un Vision Transformer

Projet de portfolio : analyser le timbre, le rythme et l’harmonie, puis mesurer leur complémentarité avec les représentations d’un Vision Transformer.

**Évaluation finale V1 terminée et vérifiée :** ViT bloc 6 + HPSS, sélectionné sur validation, atteint **macro-F1 0,5035 et accuracy 51,16 %** sur 778 morceaux test de 202 artistes. Gain apparié de macro-F1 sur ViT original : **+0,0394**, IC 95 % **[+0,0054 ; +0,0715]**. Validation : 0,5975 ; la baisse sur test reste une limite importante. Les comparaisons secondaires ne montrent pas de gain établi. [Résultats, coûts et preuves](reports/EVALUATION_FINALE.md) · [Fiche du modèle](docs/FICHE_MODELE.md) (instantané figé du 29 octobre, épinglé par les preuves V1 ; la démo locale a été livrée ensuite). Trois notebooks exécutés, démo locale et environnement utilisateur sont disponibles. Suite actuelle : 271 tests réussis (relancée le 8 novembre 2026). Session utilisateur terminée et vérifiée le 7 novembre : CNN, trente époques, meilleure époque 26, macro-F1 validation 0,5707. [Conclusion générale](docs/CONCLUSION_PROJET.md) · [Prendre en main](docs/GUIDE_LIVRAISON.md). Réinstallation réseau sur une autre machine non testée.

**Versionnage et suite :** dépôt Git local depuis le 8 novembre 2026 ; l’étiquette `v1-final` fige la V1 livrée. [Critique du projet](docs/CRITIQUE_PROJET.md) · [Plan V2](docs/PLAN_V2.md), prévu, non lancé.

**Pour reprendre rapidement :** lire [l’index des résumés](docs/resumes/INDEX.md), puis seulement la partie concernée. Chaque étape dispose d’un résumé mis à jour ; la règle est conservée dans [AGENTS.md](AGENTS.md).

## Documents de départ

- [Architecture et responsabilités](docs/ARCHITECTURE.md)
- [Brainstorming : risques, oublis et priorités](docs/BRAINSTORMING.md)
- [Plan scientifique](docs/PLAN_PROJET.md)
- [Décisions et pistes à examiner selon les résultats](docs/DECISIONS_EXPERIMENTALES.md)
- [Revue bibliographique](docs/REVUE_BIBLIOGRAPHIQUE.md)
- [Contrats des données et des résultats](docs/CONTRATS.md)
- [Environnement, GPU et benchmark reproductible](docs/ENVIRONNEMENT.md)
- [Rapport d’audit des données](reports/AUDIT_DONNEES.md)
- [Revue des anomalies et sélection initiale](reports/REVUE_DONNEES.md)
- [Reproduire les empreintes et le manifeste](docs/REPRODUCTION_REVUE.md)
- [Comprendre et reproduire les descripteurs musicaux](docs/DESCRIPTEURS_MUSICAUX.md)
- [Résultats de la première référence](reports/BASELINE_DESCRIPTEURS.md)
- [Protocole du ViT gelé](docs/VIT_GELE.md)
- [Comparaison ViT et descripteurs](reports/COMPARAISON_VIT.md)
- [Fusion : protocole et reproduction](docs/FUSION.md)
- [Résultats de la fusion et de ses ablations](reports/COMPARAISON_FUSION.md)
- [CNN : protocole, cache et reprise](docs/CNN.md)
- [Résultats et comparaison du CNN](reports/BASELINE_CNN.md)
- [Ablation des fenêtres : protocole](docs/ABLATION_FENETRES.md)
- [Résultats : une fenêtre contre trois](reports/ABLATION_FENETRES.md)
- [Bloc 6 contre bloc 12 : protocole](docs/ABLATION_BLOCS.md)
- [Résultats de l’ablation de profondeur](reports/ABLATION_BLOCS.md)
- [HPSS : protocole, analyse des entrées et figure](docs/ABLATION_HPSS.md)
- [Résultats de l’ablation des canaux](reports/ABLATION_HPSS.md)
- [Synthèse de toutes les expériences de validation](reports/SYNTHESE_VALIDATION.md)
- [Protocole des trois graines CNN](docs/REPETITIONS_CNN.md)
- [Résultats des trois graines CNN et courbes](reports/REPETITIONS_CNN.md)
- [Extension CNN : comparaison des plafonds dix et vingt](reports/EXTENSION_CNN.md)
- [Erreurs Pop/Hip-Hop et fiches d’écoute](reports/ERREURS_VALIDATION.md)
- [Utiliser la revue d’écoute et connaître ses limites](docs/ERREURS_VALIDATION.md)
- [Diagnostic audio : résultats, limites et reproduction](docs/DIAGNOSTIC_AUDIO.md)
- [Désaccords temporels : résultats sur les 780 morceaux](docs/ANALYSE_TEMPORELLE.md)
- [Combinaison bloc 6 + HPSS : protocole et faisabilité](docs/COMBINAISON_BLOC6_HPSS.md)
- [Corrections et régressions de bloc 6 + HPSS](reports/REGRESSIONS_BLOC6_HPSS.md)
- [Protocole final préparé et conditions de verrouillage](docs/PROTOCOLE_EVALUATION_FINALE.md)
- [Tâches de robustesse et d’évaluation finale](tasks/todo.md)
- [Inférence audio du candidat : utilisation et vérification](docs/INFERENCE_CANDIDAT.md)
- [Sensibilité au volume et au bruit : résultats et décision](docs/ROBUSTESSE_CANDIDAT.md)
- [Diagnostic des représentations sous bruit](reports/DIAGNOSTIC_BRUIT.md)

## Organisation

```text
configs/                  Configuration de référence et ordre des classes
src/music_genre/
  data/                   Métadonnées FMA, manifeste, audit, partitions
  audio/                  Décodage, mono, rééchantillonnage, segmentation
  features/               Log-Mel, descripteurs musicaux, embeddings, fusion
  models/                 Références, CNN et adaptateurs de Transformers
  training/               Ajustement, sélection sur validation, reprise
  evaluation/             Métriques par morceau, incertitude, robustesse
  inference/              Prédiction commune à la démo et aux scripts d’évaluation
  cli.py                  Entrée en ligne de commande ; doctor uniquement
app/                      Interface Gradio locale vérifiée
notebooks/                Analyse et fiches d’expériences pédagogiques
tests/                    Tests unitaires et contrats (271 réussis au 8 novembre)
data/                     Données locales et caches, exclus de Git
artifacts/                Poids téléchargés, checkpoints, exports
runs/                     Configuration et résultats de chaque expérience
reports/                  Synthèses, figures et tableaux de comparaison
docs/                     Bibliographie, plan, architecture et décisions
docs/resumes/             État condensé et résumé de chaque partie
scripts/                  Acquisition, inventaire et benchmark
```

## Diagnostic sans dépendances ML

Depuis la racine du projet, dans PowerShell, avec un interpréteur Python fonctionnel :

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m music_genre doctor --root .
```

Le diagnostic affiche l’interpréteur, l’architecture du processus, les versions des dépendances présentes et l’espace disque. Il ne charge pas PyTorch, ne télécharge rien et ne certifie pas la compatibilité d’un GPU.

Sur le poste actuel, le 20 octobre 2026, l’alias Windows `python` échoue. Le socle a été vérifié avec :

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& "$env:LOCALAPPDATA\Python\bin\python3.13.exe" -m music_genre doctor --root .
```

Ce Python s’exécute en x86-64 sur un système ARM64. L’environnement `.venv` a maintenant été testé avec les bibliothèques audio/ML. Le GPU Qualcomm Adreno X1-45 est présent ; son utilisation par le projet n’est pas encore validée et il ne prend pas en charge CUDA.

## Installation du socle

Après sélection de l’interpréteur, créer un environnement local :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m music_genre doctor --root .
git config core.hooksPath .githooks   # Ruff + tests avant chaque commit
```

Le package de base ne dépend que de la bibliothèque standard. Les groupes `audio`, `ml`, `dev` et `benchmark` sont installés et vérifiés sur ce poste. Matplotlib est également validé pour les figures ; Jupyter et Gradio sont également vérifiés localement dans les environnements décrits par [le guide de livraison](docs/GUIDE_LIVRAISON.md). Pour reproduire les versions CPU utilisées, suivre [la procédure détaillée](docs/ENVIRONNEMENT.md), fondée sur le relevé des dépendances installées.

## Principes

- FMA-small, huit classes d’origine, résultat principal par morceau.
- Même découpage pour tous les modèles comparés.
- Normalisations apprises sur train uniquement ; sélection sur validation.
- Test réservé aux expériences finalisées.
- Poids, données et sorties volumineuses hors du dépôt.
- Une fonction de prédiction partagée par la démo et les scripts d’évaluation ; la CLI n’a que `doctor` (commande `predict` prévue en phase 1 du plan V2).
- Les gains supposés dans le plan sont des hypothèses, pas des résultats.

**Bilan local terminé le 7 novembre :** [résultats de la session utilisateur](reports/SESSION_UTILISATEUR.md) et [conclusion](docs/CONCLUSION_PROJET.md). Aucun entraînement à relancer. La V1 reste figée ; cette expérience postérieure au test apporte un résultat de validation, sans nouvelle évaluation indépendante. Publication publique, vidéo et écoute annotée non réalisées.
