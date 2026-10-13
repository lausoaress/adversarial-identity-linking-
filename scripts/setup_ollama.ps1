# Local-model setup for Windows (PowerShell), for the "run against a
# local model first" path. Equivalent to scripts/setup_ollama.sh, which
# needs bash (WSL/Git Bash) and won't run from cmd.exe or plain PowerShell.
#
# Usage (PowerShell or cmd.exe via `powershell -File`):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_ollama.ps1 llama3.1

param(
    [string]$Model = "llama3.1"
)

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Host "Ollama not found. Install it from https://ollama.com/download (native Windows installer), then re-run this script."
    exit 1
}

Write-Host "Pulling model: $Model (this can take a while the first time)..."
ollama pull $Model
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n'ollama pull' failed - see the error above."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Done. The Windows installer runs the Ollama server automatically in the background"
Write-Host "(check the system tray icon). If 'ollama serve' says the port is already in use, that's why - it's already running."
Write-Host ""
Write-Host "Then run the demo with:       python scripts\run_demo.py"
Write-Host "Or the full experiment with:  python -m src.evaluation.run_experiment --provider ollama --model $Model"
