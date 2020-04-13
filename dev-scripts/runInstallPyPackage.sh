#!/usr/bin/env bash

# Run make and copy shared library to examples

pushd build || exit
make
# Copy .so library to examples for Python testing
cp lib/*.so ../python-package/

ls -lah lib/
popd


pip3 uninstall -y microgbtpy
python3 -m pip install --user --upgrade pip
pip3 install --user -U sklearn pandas

pushd python-package/
python setup.py bdist_wheel
pip install dist/microgbtpy*.whl --user
popd
