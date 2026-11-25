$ErrorActionPreference = "Stop"

$PythonExe = "C:\Users\黄飞\AppData\Roaming\uv\python\cpython-3.12.15-windows-x86_64-none\python.exe"
$env:PYTHONPATH = "$PSScriptRoot\.venv\Lib\site-packages;$PSScriptRoot"

Write-Host "Starting ReacherAgent Streamlit UI at http://127.0.0.1:8501 ..."
& $PythonExe -m streamlit run "$PSScriptRoot\ui\app.py" --server.port 8501 @args
exit $LASTEXITCODE
