@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py "%~dp0ui.py" %*
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    python "%~dp0ui.py" %*
    goto :end
)

echo ERROR: Python was not found on PATH.
pause

:end
endlocal
