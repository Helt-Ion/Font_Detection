@echo off

set VENV_DIR=%~dp0..\..\Pysetter\venv
set PYTHON="%VENV_DIR%\python"

set PROJECT_NAME=server

:: Put your codes here

cd %PROJECT_NAME%
%PYTHON% manage.py runserver

@echo on
pause
