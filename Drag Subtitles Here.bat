@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo Drag subtitle files ^(.srt .vtt .ass .ssa^) OR videos with soft tracks ^(.mkv .mp4 ...^) onto this file.
    echo Or double-click to type a path instead.
    echo.
)

where py >nul 2>nul
if %errorlevel%==0 (
    py "%~dp0dual_subs.py" %*
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    python "%~dp0dual_subs.py" %*
    goto :end
)

echo ERROR: Python was not found on PATH. Install Python from https://python.org and try again.
pause

:end
endlocal
