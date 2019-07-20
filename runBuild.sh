#!/usr/bin/env bash

# Run cmake (downloads gtest on every invocation)

# Clone pybind11 repository as GIT submodule
if [[ ! -d pybind11 ]]; then	
  git submodule add -f https://github.com/pybind/pybind11.git
  git submodule update --init --recursive
fi


rm -rf bin/
rm -rf build/
mkdir build/
pushd build || exit
cmake ..
popd || exit

./runMake.sh