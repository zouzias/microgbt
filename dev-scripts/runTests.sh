#!/usr/bin/env bash

set -o pipefail

echo "***************************"
echo "***************************"
echo "*      C++ tests          *"
echo "***************************"
echo "***************************"
pushd build
make
./bin/unit_tests