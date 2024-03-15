#!/bin/bash

export VENV_DIR=venv
export PYTHON="$PWD/../../../../Pysetter/$VENV_DIR/bin/python3.10"

$PYTHON -m src.font_recognize

read
