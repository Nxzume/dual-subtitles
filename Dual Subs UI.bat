@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Virtual environment not found.
    echo Run setup.bat first ^(creates .venv and installs requirements^).
    echo.
    pause
    exit /b 1
)

"%VENV_PY%" "%~dp0dual_subs.py" --ui %*
if errorlevel 1 (
    echo.
    echo App exited with an error.
    pause
    exit /b 1
)
endlocal
