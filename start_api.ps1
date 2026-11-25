$ErrorActionPreference = "Stop"

$PythonExe = "C:\Users\黄飞\AppData\Roaming\uv\python\cpython-3.12.15-windows-x86_64-none\python.exe"
$env:PYTHONPATH = "$PSScriptRoot\.venv\Lib\site-packages;$PSScriptRoot"

Write-Host "Starting ReacherAgent Web & API server at http://127.0.0.1:8000 ..."
Write-Host "Open http://127.0.0.1:8000 in your browser (built frontend + REST/SSE API)."
& $PythonExe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 @args
exit $LASTEXITCODE
