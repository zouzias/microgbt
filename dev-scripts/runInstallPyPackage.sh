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
python3 -m pip install --user -U sklearn pandas wheel auditwheel


pushd python-package/
echo "Building wheel..."
rm -rf dist/
python3 setup.py bdist_wheel

echo "Repairing wheel by injecting shared library file"
auditwheel repair dist/microgbtpy*.whl

echo "Installing wheel..."
pip3 install dist/microgbtpy*.whl -v
popd
