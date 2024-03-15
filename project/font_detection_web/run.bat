@echo off

set VENV_DIR=%~dp0..\..\Pysetter\venv
set PYTHON="%VENV_DIR%\python"

set PROJECT_NAME=server
set PORT=9000

:: Put your codes here

cd %PROJECT_NAME%
%PYTHON% manage.py runserver %PORT%

@echo on
pause
