#!/usr/bin/env bash

echo "***************************"
echo "***************************"
echo "*      Python tests       *"
echo "***************************"
echo "***************************"

pushd python-package
python3 -m pytest
popd