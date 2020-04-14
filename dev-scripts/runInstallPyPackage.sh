#!/usr/bin/env bash

# Run make and copy shared library to examples

pushd build || exit
make
# Copy .so library to examples for Python testing
cp lib/*.so ../python-package/

ls -lah lib/
popd


python3 -m pip install --user --upgrade pip
python3 -m pip uninstall -y microgbtpy
python3 -m pip3 install --user -U sklearn pandas wheel

pushd python-package/
python3 setup.py bdist_wheel
python3 -m pip install dist/microgbtpy*.whl --user
popd
