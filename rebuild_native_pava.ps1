Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Join-Path $workspace "sksfolio-0.2.0\sksfolio-0.2.0"
$python = Join-Path $workspace ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtualenv interpreter not found at $python"
}

if (-not (Test-Path $project)) {
    throw "Project root not found at $project"
}

if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    throw "cl.exe is not in PATH. Run this from a Visual Studio Developer PowerShell after installing Build Tools."
}

Push-Location $project
try {
    & $python -m pip install -e . --force-reinstall --no-build-isolation
    if ($LASTEXITCODE -ne 0) {
        throw "pip reinstall failed"
    }
}
finally {
    Pop-Location
}
