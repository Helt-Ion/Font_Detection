@echo off

set VENV_DIR=%~dp0..\..\Pysetter\venv
set PYTHON="%VENV_DIR%\python"

:: Put your codes here

%PYTHON% mynet_test.py

@echo on
pause
