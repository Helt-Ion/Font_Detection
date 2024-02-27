#!/bin/bash

# export INSTALL_TORCH=
# export TORCH_INDEX_URL=
# export TORCH_PACKAGE=
# export TORCHVISION_PACKAGE=
export INSTALL_XFORMERS=False
# export XFORMERS_PACKAGE=
export INSTALL_ACCELERATE=False
# export ACCELERATE_PACKAGE=
# export REQS_FILE=

export PYTHON_BASE=Python310
export PYTHON_BASE_FULL="$PWD/$PYTHON_BASE/bin/python3.10"
export VENV_DIR=venv
export PYTHON="$PWD/$VENV_DIR/bin/python3.10"

if [ ! -d $VENV_DIR ]; then
  echo Creating venv...
  cp -r $PYTHON_BASE $VENV_DIR
fi

$PYTHON build.py

