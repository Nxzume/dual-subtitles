@echo off
setlocal
cd /d "%~dp0"

set "VENV=%~dp0.venv"
set "PY="

where py >nul 2>nul
if %errorlevel%==0 set "PY=py -3"

if not defined PY (
    where python >nul 2>nul
    if %errorlevel%==0 set "PY=python"
)

if not defined PY (
    echo ERROR: Python was not found on PATH.
    echo Install Python 3.10+ from https://python.org and try again.
    pause
    exit /b 1
)

echo Using: %PY%
echo Creating virtual environment in .venv ...
%PY% -m venv "%VENV%"
if errorlevel 1 (
    echo ERROR: failed to create .venv
    pause
    exit /b 1
)

echo Installing requirements ...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)

if not exist "%~dp0.env" (
    if exist "%~dp0.env.example" (
        copy "%~dp0.env.example" "%~dp0.env" >nul
        echo Created .env from .env.example — edit it and set NVIDIA_API_KEY.
    )
) else (
    echo .env already exists — left unchanged.
)

REM Shortcut with CC icon (Explorer shows bat icon otherwise)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$dir = '%~dp0'.TrimEnd('\'); $ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut((Join-Path $dir 'Dual Subs UI.lnk')); $s.TargetPath = Join-Path $dir 'Dual Subs UI.bat'; $s.WorkingDirectory = $dir; $ico = Join-Path $dir 'assets\app.ico'; if (Test-Path $ico) { $s.IconLocation = $ico + ',0' }; $s.Description = 'Dual Subtitles'; $s.Save()"
if exist "%~dp0Dual Subs UI.lnk" (
    echo Created "Dual Subs UI.lnk" ^(CC icon^).
)

echo.
echo Setup complete. Double-click "Dual Subs UI.lnk" or "Dual Subs UI.bat" to open the app.
pause
endlocal
