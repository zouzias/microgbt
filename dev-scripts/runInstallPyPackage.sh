#!/usr/bin/env bash

python3 -m pip install --user --upgrade pip
python3 -m pip uninstall -y microgbtpy
python3 -m pip install --user -U sklearn pandas numpu

echo "Installing package..."
pip3 install .
