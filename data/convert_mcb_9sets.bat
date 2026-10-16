@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title MCB - Generate 9 Point Cloud Datasets
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
rem 0 = automatically choose workers based on CPU and available memory.
set "MCB_WORKERS=0"
if not exist "%~dp0MCB_merged\train\" goto missing_source
if not exist "%~dp0MCB_merged\test\" goto missing_source
if not exist "%~dp0_pointcloud_converter\bootstrap.ps1" goto missing_helper
if not exist "%~dp0_pointcloud_converter\convert_mcb.py" goto missing_helper
if not exist "%~dp0_pointcloud_converter\sampling_windows_x64.dll" goto missing_helper
echo Converting MCB_merged into 9 float32 XYZ NPY datasets.
echo First launch downloads Python and NumPy. Internet is required once.
echo Completed files are verified and resumed on subsequent launches.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0_pointcloud_converter\bootstrap.ps1" -DataRoot "%~dp0."
set "MCB_EXIT=%ERRORLEVEL%"
echo.
if "%MCB_EXIT%"=="0" (echo Conversion completed successfully.) else (echo Conversion stopped. See the error above; rerun to resume.)
echo Press any key to close this window.
pause >nul
exit /b %MCB_EXIT%
:missing_source
echo ERROR: Put MCB_merged beside this BAT, with its train and test folders.
pause
exit /b 1
:missing_helper
echo ERROR: Copy the entire _pointcloud_converter folder beside this BAT.
pause
exit /b 1
