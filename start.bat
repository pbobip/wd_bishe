@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "PYTHON_EXE=%USERPROFILE%\fy\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo [wd_bishe] Frontend will start in a new window...
start "wd_bishe_frontend" powershell -NoExit -Command "Set-Location '%ROOT%\frontend'; npm run dev"

echo [wd_bishe] Backend starting in current window...
cd /d "%ROOT%"
"%PYTHON_EXE%" -m uvicorn backend.main:app --reload --app-dir "%ROOT%" --reload-dir "%ROOT%\backend"

endlocal
