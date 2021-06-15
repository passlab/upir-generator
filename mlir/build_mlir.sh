#!/bin/bash

BUILD_DIR=$HOME/Projects/mlir/llvm-project/build
PREFIX=$HOME/Projects/mlir/llvm-project/build
REX_ROOT=$HOME/Projects/rexdev
mkdir build
cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit -DREX_INSTALL=$REX_ROOT/rex_install
cmake --build .
