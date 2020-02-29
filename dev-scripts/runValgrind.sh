#!/usr/bin/env bash

echo "***************************"
echo "***************************"
echo "*      Valgrind           *"
echo "***************************"
echo "***************************"
pushd build
valgrind --leak-check=full --track-fds=yes --track-origins=yes --leak-check=full --show-leak-kinds=all --error-exitcode=1 ./bin/unit_tests
popd