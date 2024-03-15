@echo off

set VENV_DIR=%~dp0..\..\Pysetter\venv
set PYTHON="%VENV_DIR%\python"

:: Put your codes here

%PYTHON% -m run.font_recognize

@echo on
pause
