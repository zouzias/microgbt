#!/usr/bin/env bash

# Run make and copy shared library to examples

pushd build || exit
make
# Copy .so library to examples for Python testing
cp lib/*.so ../python-package/

ls -lah lib/
popd


python3 -m pip install --user --upgrade pip

pushd python-package/
pip3 install --user sklearn pandas
python3 setup.py build_ext install --user
popd
