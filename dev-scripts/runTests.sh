#!/usr/bin/env bash

echo "***************************"
echo "***************************"
echo "*      C++ tests          *"
echo "***************************"
echo "***************************"
pushd build
make
./bin/unit_tests
popd

echo "***************************"
echo "***************************"
echo "*      Python tests       *"
echo "***************************"
echo "***************************"

pushd python-package
pytest
popd