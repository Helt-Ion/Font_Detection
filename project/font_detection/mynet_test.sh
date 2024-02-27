#!/bin/bash

export VENV_DIR=venv
export PYTHON="$PWD/../../Pysetter/$VENV_DIR/bin/python3.10"

$PYTHON mynet_test.py

read
