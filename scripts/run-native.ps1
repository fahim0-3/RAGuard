<#
.SYNOPSIS
Runs RAGuard on Windows without Docker.

.DESCRIPTION
The API and Streamlit UI run in the local Python virtual environment. PostgreSQL
is intentionally not started here: set DATABASE_URL in .env to a managed
PostgreSQL instance with the pgvector extension enabled.

Model downloads are kept beneath .cache\raguard-models in the repository,
rather than the default per-user Hugging Face cache on the C: drive.
#>
[CmdletBinding()]
param(
    [ValidateSet("api", "frontend", "ingest", "setup-db", "check-db")]
    [string]$Task = "api",

    [switch]$Reset,

    [switch]$OfflineModels,

    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found. Run: uv sync --locked --all-groups"
}

# These process-level variables take precedence over .env and keep Hugging
# Face/SentenceTransformer caches in the project rather than %USERPROFILE% on C:.
#
# The project cache is used only when it actually holds a model. Pointing at an
# empty directory while LOCAL_MODEL_OFFLINE=true is the worst of both worlds:
# downloads are blocked and there is nothing to load, so start-up fails with a
# Hugging Face connection error that says nothing about the real cause. Falling
# back to the default cache uses models already on disk instead of re-fetching
# several gigabytes.
$modelCache = Join-Path $projectRoot ".cache\raguard-models"
$projectHub = Join-Path $modelCache "hub"
$hasProjectModels = (Test-Path -LiteralPath $projectHub) -and
    @(Get-ChildItem -LiteralPath $projectHub -Filter "models--*" -Directory -ErrorAction SilentlyContinue).Count -gt 0

if ($hasProjectModels) {
    $env:HF_HOME = $modelCache
    $env:HF_HUB_CACHE = $projectHub
    $env:TRANSFORMERS_CACHE = Join-Path $modelCache "transformers"
    $env:SENTENCE_TRANSFORMERS_HOME = Join-Path $modelCache "sentence-transformers"
    Write-Host "Model cache: $modelCache"
} else {
    $defaultHub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
    if (Test-Path -LiteralPath $defaultHub) {
        Write-Host "Model cache: $defaultHub (the project cache holds no models)"
    } else {
        # Neither cache has anything. Send downloads to the project so a first
        # run still keeps its models off C:, and let the load report the miss.
        $env:HF_HOME = $modelCache
        $env:HF_HUB_CACHE = $projectHub
        $env:TRANSFORMERS_CACHE = Join-Path $modelCache "transformers"
        $env:SENTENCE_TRANSFORMERS_HOME = Join-Path $modelCache "sentence-transformers"
        Write-Host "Model cache: $modelCache (empty; models will be downloaded unless offline)"
    }
}
# The Xet transport can hang on some Windows/network combinations before it
# writes a single byte. Plain HTTPS supports resumable downloads and is more
# reliable for this native launcher.
$env:HF_HUB_DISABLE_XET = "1"
# Transformers otherwise starts a background download of a second safetensors
# copy when a repository ships only PyTorch weights. RAGuard already loaded the
# trusted upstream weights, so that duplicate is unnecessary local storage.
$env:DISABLE_SAFETENSORS_CONVERSION = "1"
if ($OfflineModels) {
    # Fail fast instead of contacting Hugging Face. Use only after one
    # successful online start has populated the repository-local cache.
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
}
# Select the validated application profile instead of rewriting individual
# model settings. Docker and production keep the default `full` profile.
$env:RAGUARD_RUNTIME_PROFILE = "local_compact"
$env:ADMISSION_BACKEND = "local"

if ($Task -ne "check-db") {
    # Only the paths this launcher set: when it falls back to the default cache
    # these are unset, and the Hugging Face libraries manage that location.
    $cacheDirs = @($env:HF_HUB_CACHE, $env:TRANSFORMERS_CACHE, $env:SENTENCE_TRANSFORMERS_HOME) |
        Where-Object { $_ }
    if ($cacheDirs) {
        New-Item -ItemType Directory -Force -Path $cacheDirs | Out-Null
    }
}

Push-Location $projectRoot
$processExitCode = 0
try {
    switch ($Task) {
        "api" {
            $arguments = @("-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000")
            if ($Reload) {
                $arguments += "--reload"
            }
            & $python @arguments
            $processExitCode = $LASTEXITCODE
        }
        "frontend" {
            & $python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
            $processExitCode = $LASTEXITCODE
        }
        "ingest" {
            $arguments = @("-m", "src.ingestion.ingest")
            if ($Reset) {
                $arguments += "--reset"
            }
            & $python @arguments
            $processExitCode = $LASTEXITCODE
        }
        "setup-db" {
            & $python -m scripts.setup_remote_db
            $processExitCode = $LASTEXITCODE
        }
        "check-db" {
            & $python -m scripts.check_remote_db
            $processExitCode = $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}

exit $processExitCode
