$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
Write-Host 'Demo locale : http://127.0.0.1:7860 ; fermer avec Ctrl+C.'
& (Join-Path $projectRoot '.venv-demo/Scripts/python.exe') (Join-Path $projectRoot 'app/main.py')
exit $LASTEXITCODE
