@echo off

set VENV_DIR=%~dp0..\..\Pysetter\venv
set PYTHON="%VENV_DIR%\python"

:: Put your codes here

%PYTHON% -m MyNet.test

@echo on
pause
