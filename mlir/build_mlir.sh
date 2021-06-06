#!/bin/bash

BUILD_DIR=$HOME/Projects/mlir/llvm-project/build
PREFIX=$HOME/Projects/mlir/llvm-project/build
mkdir build
cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone
