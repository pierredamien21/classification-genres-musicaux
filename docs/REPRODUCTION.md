# Reproduction

Mis à jour : 11 novembre 2026. Commandes PowerShell, lancées depuis la racine. La procédure V1 détaillée reste dans [archive/ENVIRONNEMENT.md](archive/ENVIRONNEMENT.md).

## 1. Environnement

Prérequis : `uv` et Python 3.13 x86-64. Le poste de référence est un Windows ARM64 qui l'exécute en émulation, sans CUDA.

```powershell
uv venv --python "$env:LOCALAPPDATA\Python\bin\python3.13.exe" --no-python-downloads .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend cpu -r requirements/windows-cpu-py313.txt
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
uv pip check --python .venv\Scripts\python.exe
git config core.hooksPath .githooks   # Ruff + tests avant chaque commit
```

La démo et les notebooks ont leurs propres fichiers dans `requirements/`.

**Vérifié le 11 novembre 2026** sur ce poste, à partir d'un clone propre et d'un cache vide :
- 76 paquets compatibles installés en 97 s, soit environ 1,1 Go de cache ;
- Ruff réussi ;
- **241 tests réussis, 68 ignorés**, faute de données, de poids et d'exports locaux, avec la raison affichée.

Sur le poste complet, les 309 tests passent.

**Non vérifié** : un autre compte, un autre poste, un Python ARM64 natif.

## 2. Données (jamais versionnées)

Placer `fma_metadata` et `fma_small` dans `data/raw/`, en suivant la procédure V1 : `scripts/acquire_fma.py`, puis `audit_fma.py` et `build_baseline_manifest.py`.

Le développement V2 réutilise ces 7 857 morceaux. L'audio V2 est relu par plages dans `fma_large.zip`, puis audité :

```powershell
.\.venv\Scripts\python.exe scripts/fetch_v2_audio.py
.\.venv\Scripts\python.exe scripts/audit_v2_cohorts.py
```

Les listes, les empreintes et les règles de remplacement sont figées par le [verrou 2](../reports/v2/lock2.json). Une relance doit retrouver les mêmes empreintes SHA-256.

## 3. Poids préentraînés (jamais versionnés)

```powershell
.\.venv\Scripts\python.exe scripts/fetch_vit.py
.\.venv\Scripts\python.exe scripts/fetch_audio_models.py --model maest   # aussi : ast, panns_cnn14_16k
```

Chaque fichier est contrôlé par la révision et la SHA-256 de sa configuration. `--verify-only` contrôle sans accès au réseau.

| Poids | Licence |
|---|---|
| ViT tiny (timm) | Apache-2.0 |
| AST | BSD-3-Clause |
| PANNs CNN14 | CC-BY-4.0 |
| MAEST | CC-BY-NC-SA-4.0, non commercial |

## 4. Chaîne V2, dans l'ordre

| Étape | Commande (exemple) | Coût mesuré sur ce poste |
|---|---|---|
| Caractéristiques | `extract_features.py --config configs/maest.json --development --workers 7` | environ 2,5 à 3,8 h |
| Validation croisée | `cv_v2.py --features dev_maest --family layer12 --threads 8` | quelques minutes |
| Tableau CV | `summarize_cv_v2.py` | secondes |
| CNN renforcé | `train_cnn.py --config configs/cnn_v2.json --seed 42` (puis 43 et 44) | environ 1 h par graine |
| Dégradations | `extract_features.py ... --degradation snr10` (configs/degradations_v2.json) | environ 2,5 h chacune |
| Robustesse, calibration | `robustness_v2.py --run ...`, puis `calibrate_v2.py --run ... --robustness ...` | quelques minutes |
| Tête finale | `fit_final_v2.py --run runs/v2-cv/dev_maest-layer12-...` | quelques minutes |
| Répétition | `evaluate_v2.py --rehearsal 3 --workers 7` | environ 2 min |

Toutes les commandes se lancent avec `.\.venv\Scripts\python.exe scripts/<nom>`.

**L'évaluation finale ne se relance pas.** `evaluate_v2.py --final` a tourné une fois, le 11 novembre, sous le verrou 3. Il refuse désormais de tourner : son rapport existe, et le moindre fichier ou paquet modifié est détecté. La reproduction des chiffres passe par les prédictions archivées :

```powershell
.\.venv\Scripts\python.exe scripts/recount_v2.py --evaluation reports/v2/final_evaluation
.\.venv\Scripts\python.exe scripts/interpret_v2.py
```

Le recomptage doit retrouver les points à 10⁻⁹ près. Vérifié le 11 novembre : `status: agrees`, écart nul.

Il réécrit `recount.json` dans le dossier indiqué, dont seul le chemin enregistré change. Pour garder l'archive intacte, restaurer ensuite le fichier avec `git checkout -- reports/v2/final_evaluation/recount.json`.

`interpret_v2.py` crée aussi une nouvelle fiche d'écoute dans `data/interim/`, si l'audio est présent.

## 5. Ce qu'une relance ne reproduira pas à l'identique

- **Durées** : elles dépendent du poste.
- **Résultats bit à bit** : les extractions ont été contrôlées par répétition, sans écart pour les 10 systèmes. Une autre machine ou une autre version de BLAS peut toutefois introduire de très petits écarts numériques.
- **Annotations d'écoute** : elles sont humaines, archivées dans `reports/v2/listening/`, et ne se recalculent pas.
