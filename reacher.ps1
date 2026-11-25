$ErrorActionPreference = "Stop"

$PythonExe = "C:\Users\黄飞\AppData\Roaming\uv\python\cpython-3.12.15-windows-x86_64-none\python.exe"
$env:PYTHONPATH = "$PSScriptRoot\.venv\Lib\site-packages;$PSScriptRoot"

& $PythonExe -m core.cli @args
exit $LASTEXITCODE
