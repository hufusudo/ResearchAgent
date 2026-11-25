$ErrorActionPreference = "Stop"

Push-Location "$PSScriptRoot\frontend"
try {
    Write-Host "Starting ReacherAgent Vite Dev Server at http://127.0.0.1:5173 ..."
    npm run dev -- --host 127.0.0.1
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
