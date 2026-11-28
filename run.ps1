# Ael-Agents Launcher for Windows (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Ael-Agents // Horizon & Amnesia Breaker        " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "[✓] uv detected. Syncing virtual environment..." -ForegroundColor Green
    uv sync
    uv run python main.py @args
} else {
    Write-Host "[!] uv not detected. Falling back to Python venv..." -ForegroundColor Yellow
    if (-not (Test-Path ".venv")) {
        Write-Host "[+] Creating virtual environment..." -ForegroundColor Cyan
        python -m venv .venv
        .\.venv\Scripts\pip install -e .
    }
    .\.venv\Scripts\python main.py @args
}
