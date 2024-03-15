#!/bin/bash

export VENV_DIR=venv
export PYTHON="$PWD/../../Pysetter/$VENV_DIR/bin/python3.10"

export PROJECT_NAME=server

cd $PROJECT_NAME
$PYTHON -m manage migrate

read
