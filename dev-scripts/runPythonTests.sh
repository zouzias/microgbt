#!/usr/bin/env bash

set -o pipefail

echo "***************************"
echo "***************************"
echo "*      Python tests       *"
echo "***************************"
echo "***************************"

python3 -m pytest python-package/tests/
