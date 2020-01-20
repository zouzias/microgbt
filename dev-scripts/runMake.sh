#!/usr/bin/env bash

set -x
# Run make to build static cpp library and install Python package


pushd build || exit
make
ls -lah lib/
popd


./dev-scripts/runInstallPyPackage.sh
