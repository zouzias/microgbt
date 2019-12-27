#!/usr/bin/env bash

# Run make and copy shared library to examples

./runInstallPyPackage.sh

# Run titanic example
pushd ./examples/
./test-titanic.py
popd
