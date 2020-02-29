#!/usr/bin/env bash

echo "***************************"
echo "***************************"
echo "*      Python tests       *"
echo "***************************"
echo "***************************"

pushd python-package
pytest
popd