#!/bin/bash

export PYTHON_BASE=Python310
export PYTHON_BASE_FULL="$PWD/$PYTHON_BASE/bin/python3.10"

echo Resetting git proxy...

git config --global --unset http.proxy

echo Resetting pip proxy...

$PYTHON_BASE_FULL -m pip config unset global.proxy
