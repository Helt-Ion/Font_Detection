#!/bin/bash

export PYTHON_BASE=Python310
export PYTHON_BASE_FULL="$PWD/$PYTHON_BASE/bin/python3.10"

$PYTHON_BASE_FULL -m pip config unset global.index-url

read
