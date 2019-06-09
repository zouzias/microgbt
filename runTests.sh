#!/usr/bin/env bash

# Run tests

pushd build
make
./bin/unit_tests
popd