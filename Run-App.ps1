# Smart Job Description Analyzer — local run (avoids Program Files permission errors).
# Usage: right-click → Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File .\Run-App.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment in .venv ..."
    python -m venv .venv
}

& ".\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip and installing dependencies ..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Starting Streamlit (http://localhost:8501) ..."
python -m streamlit run app.py
