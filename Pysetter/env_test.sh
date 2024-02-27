#!/bin/bash

export VENV_DIR=venv
export PYTHON="$PWD/$VENV_DIR/bin/python3.10"

if [ ! -d $VENV_DIR ]; then
  echo venv does not exist, skipping.
  exit 0
fi

$PYTHON env_test.py

