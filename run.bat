@echo off
setlocal
if not defined WATER_PYTHON set "WATER_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%WATER_PYTHON%" (
    echo Python environment not found: "%WATER_PYTHON%" 1>&2
    echo Run: py -3.13 "%~dp0scripts\setup_env.py" --check 1>&2
    exit /b 1
)
"%WATER_PYTHON%" "%~dp0app.py" %*
exit /b %ERRORLEVEL%
