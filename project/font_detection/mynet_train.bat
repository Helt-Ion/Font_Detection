@echo off

set VENV_DIR=%~dp0..\..\Pysetter\venv
set PYTHON="%VENV_DIR%\python"

:: Put your codes here

%PYTHON% -m MyNet.train

@echo on
pause
