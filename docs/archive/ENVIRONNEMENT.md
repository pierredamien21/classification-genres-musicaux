# Environnement audio et premier benchmark

Vérifié le 20 octobre 2026. Python 3.13.16 x86-64 dans `.venv`, sur le poste Windows ARM64.
Les versions effectivement installées sont conservées dans
[`requirements/windows-cpu-py313.txt`](../../requirements/windows-cpu-py313.txt).
Ce relevé concerne cet environnement précis ; il ne garantit pas les autres plateformes.

## Matériel et accélération

Le registre Windows identifie **Qualcomm(R) Adreno(TM) X1-45 GPU**, pilote 31.0.152.0.
Un GPU est donc présent. Ce GPU Qualcomm ne prend pas en charge CUDA.
PyTorch 2.14.1+cpu a été installé ; `torch.version.cuda` vaut `None`.
Ce test PyTorch seul ne permettrait pas de conclure à l’absence de GPU.

Microsoft documente l’accélération de GPU Qualcomm via
[DirectML](https://github.com/microsoft/DirectML). Cette bibliothèque est désormais en
maintenance et Microsoft oriente les nouveaux usages vers Windows ML.
La [voie PyTorch avec DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows)
existe, mais sa compatibilité avec notre Python, nos versions et toutes les opérations
du ViT n’a pas été testée. Ne pas assimiler disponibilité matérielle, exécution effective
sur GPU et gain de vitesse. Toute expérience GPU utilisera un environnement séparé,
une comparaison numérique des embeddings et des mesures après initialisation.

Le benchmark actuel est exclusivement CPU, avec quatre threads PyTorch.

## Reproduire sous PowerShell

Depuis la racine, avec `uv` disponible et le Python x86-64 indiqué installé :

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location).Path 'data\interim\uv-cache'
uv venv --python "$env:LOCALAPPDATA\Python\bin\python3.13.exe" --no-python-downloads .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend cpu -r requirements/windows-cpu-py313.txt
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
uv pip check --python .venv\Scripts\python.exe
.\.venv\Scripts\python.exe scripts/fetch_vit.py
.\.venv\Scripts\python.exe scripts/benchmark_pipeline.py
```

La création de l’environnement est destinée à une installation neuve. Les données
FMA-small doivent déjà être acquises. Le téléchargement du modèle utilise une révision
fixe et des SHA-256 attendus dans `configs/vit_checkpoint.json`. Le traitement normal
ne télécharge jamais de poids implicitement. `fetch_vit.py --verify-only` contrôle les
fichiers déjà présents sans accès réseau.

Pour les tests, choisir un dossier temporaire **neuf** dans le projet ; pytest peut
effacer le contenu de `--basetemp` :

```powershell
$testDirectory = Join-Path (Get-Location).Path ('data\interim\pytest-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest --basetemp $testDirectory -q
.\.venv\Scripts\ruff.exe check src tests scripts
```

## Ce qui est réellement mesuré

Complément du 21 octobre 2026 : Matplotlib 3.11.2 est installé pour les figures
HPSS ; six paquets, Matplotlib compris, sont ajoutés au relevé des dépendances.
Les versions audio/ML du benchmark sont inchangées, vérifiées par le diagnostic.
`uv pip check` confirme la compatibilité des 67 paquets installés. La génération
locale des figures PNG/PDF est vérifiée ; Jupyter et Gradio restent à valider.

Sélection déterministe de 64 morceaux de **training**, huit par genre, graine 42.
Chaque MP3 est décodé, moyenné en mono, rééchantillonné à 16 kHz ; une fenêtre de
10 secondes commençant à 10 secondes est retenue. Log-Mel 128 bandes, FFT 1024,
pas 256, puis dB de puissance avec référence fixe 1.

Le diagnostic utilise un intervalle fixe de −80 à +20 dB, un redimensionnement complet
à 224 × 224, trois canaux identiques et la normalisation du checkpoint. La moyenne
des tokens de patches du ViT-Tiny gelé produit 192 valeurs. Le token de classe est exclu.
Le redimensionnement sans recadrage est une adaptation à l’audio, pas le prétraitement
image complet du modèle original. L’intervalle dB et la représentation restent à étudier
sur training/validation ; aucune optimisation n’a utilisé le test.

Résultat : 64 sorties finies, médiane de 0,112 s par morceau traité, dont 0,027 s pour
le ViT. Les temps excluent imports, chargement du modèle et initialisation.
Il s’agit d’une faisabilité d’inférence CPU ; aucun classifieur de genres n’a été entraîné.
Voir le [rapport détaillé](../../reports/environment_benchmark.json).

Une alerte native mpg123 (`part2_3_length ... too large for available bit count`) est
apparue malgré les 64 sorties. Le benchmark ne capture pas ces alertes par fichier :
le statut `passed` signifie que les fonctions ont retourné des sorties valides, pas
que les flux MP3 sont exempts de défauts. L’audit intégral devra les rattacher aux morceaux,
documenter les durées, les silences et les éventuelles exclusions avant entraînement.
