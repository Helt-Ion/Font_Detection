@echo off

set VENV_DIR=%~dp0..\..\Pysetter\venv
set PYTHON="%VENV_DIR%\python"
set DJANGO_ADMIN="%VENV_DIR%\Scripts\django-admin.exe"

set PROJECT_NAME=server

:: Put your codes here

%DJANGO_ADMIN% startproject %PROJECT_NAME%

@echo on
pause
