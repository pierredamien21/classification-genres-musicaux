$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:JUPYTER_RUNTIME_DIR = Join-Path $projectRoot 'data/interim/jupyter-runtime'
$env:IPYTHONDIR = Join-Path $projectRoot 'data/interim/ipython'
& (Join-Path $projectRoot '.venv-notebook/Scripts/python.exe') -m jupyterlab --ip=127.0.0.1 --ServerApp.root_dir="$projectRoot"
exit $LASTEXITCODE
