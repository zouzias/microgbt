#!/usr/bin/env bash

set -o pipefail

echo "***************************"
echo "***************************"
echo "*      Python tests       *"
echo "***************************"
echo "***************************"

pushd python-package/
python3 -m pytest tests/
