#!/usr/bin/env bash

# Run make and copy shared library to examples

pushd build || exit
make
# Copy .so library to examples for Python testing
cp lib/*.so ../python-package/

ls -lah lib/
popd

pushd python-package/
pip3 install sklearn pandas
pip3 install -U .
popd
