#!/usr/bin/env bash

# Run make and copy shared library to examples

pushd build || exit
make
# Copy .so library to examples for Python testing
cp lib/*.so ../python-package/
popd

pushd ./python-package/
pip install sklearn pandas
pip install -U .
popd

# Run titanic example
pushd ./examples/
./test-lightgbm-example.py
popd
