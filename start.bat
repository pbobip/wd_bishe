@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

powershell -ExecutionPolicy Bypass -File "%ROOT%\scripts\dev-up.ps1"

endlocal
