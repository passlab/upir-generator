#!/bin/bash

REX_ROOT=$HOME/Projects/rexdev
mkdir build
cd build
cmake -G Ninja .. -DREX_INSTALL=$REX_ROOT/rex_install
cmake --build .
