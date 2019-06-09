#!/usr/bin/env bash

# Run make and copy shared library to examples

pushd build
make
# Copy .so library to examples for Python testing
cp lib/*.so ../examples/
popd
