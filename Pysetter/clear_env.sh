#!/bin/bash

export VENV_DIR=venv

if [ -d $VENV_DIR ]; then
  echo Clearing venv...
  rm -rf $VENV_DIR
else
  echo venv does not exist, skipping.
  read
  exit 0
fi

echo venv cleared.

read
