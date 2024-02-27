#!/bin/bash

export PYTHON_BASE=Python310
export PYTHON_BASE_FULL="$PWD/$PYTHON_BASE/bin/python3.10"

echo Setting git proxy...

git config --global http.proxy "http://127.0.0.1:7890"

echo Setting pip proxy...

$PYTHON_BASE_FULL -m pip config set global.proxy http://127.0.0.1:7890

read

