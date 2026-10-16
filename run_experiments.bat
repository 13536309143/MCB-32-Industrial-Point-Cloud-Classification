@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "MCB_PROFILE=%~1"
if not defined MCB_PROFILE set "MCB_PROFILE=full"
echo MCB industrial model classification - PointNeXt / PointNet / PointNet++
echo Profile: %MCB_PROFILE%
echo Default FULL profile: 1296 training runs, 11898 test jobs. This is a long experiment suite.
echo First setup needs internet. WSL installation may request administrator access or a reboot.
echo Usage: run_experiments.bat [full ^| quick ^| smoke]
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0experiments\bootstrap_windows.ps1" -ProjectRoot "%~dp0." -Profile "%MCB_PROFILE%"
set "MCB_EXIT=%ERRORLEVEL%"
echo.
if "%MCB_EXIT%"=="0" (echo All requested experiments completed.) else (echo Stopped with exit code %MCB_EXIT%. Read the error above and the result logs.)
pause
exit /b %MCB_EXIT%
