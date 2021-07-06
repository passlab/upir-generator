# An out-of-tree MLIR dialect

This is an example of an out-of-tree [MLIR](https://mlir.llvm.org/) dialect for Parallel IR.

## Workflow

1. REX compiler parses the input source code and constructs a Sage AST.
2. Optionally, we can perform some optimizations that won't eliminate any parallel information.
3. By traversing the Sage AST, an MLIR AST using multiple dialects is created. The core part is `parallel` dialect, which converts the input OpenMP code to a language-neutral parallel IR.
5. Based on their needs or toolchain, users can either use Sage AST or MLIR AST to perform optimization, transformation and the rest compiling work.

MLIR libraries are required but not Clang.

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```bash
export BUILD_DIR=$HOME/Projects/mlir/llvm-project/build
export PREFIX=$HOME/Projects/mlir/llvm-project/build
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build .
```
To build the documentation from the TableGen description of the dialect operations, run
```bash
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## Running

`rex2mlir` creates an MLIR AST for the following simple program and print out the AST.

```c
void axpy (float* x, float* y, float a, int n) {
    int i;
#pragma omp parallel for num_threads(6)
    for (i = 0; i < n; i++) {
        y[i] = y[i] + a * x[i];
    }
}
```

In this example, three MLIR dialects are used, which are [`std`](https://mlir.llvm.org/docs/Dialects/Standard/), [`scf`](https://mlir.llvm.org/docs/Dialects/SCFDialect/)([structured control flow](https://llvm.discourse.group/t/rfc-rename-loopops-dialect-to-scf-structured-control-flow/872)), and `parallel`.
`parallel` dialect can be converted into [`scf`](https://mlir.llvm.org/docs/Dialects/SCFDialect/), [`omp`](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/), [`acc`](https://mlir.llvm.org/docs/Dialects/OpenACCDialect/), [`llvm`](https://mlir.llvm.org/docs/Dialects/LLVM/) or other suitable dialects.

The output should be:

```bash
Set up MLIR environment....
Prepare a dummy code location....
Prepare base function parameters....
Create a base function....
Create the body of base function....
Insert a SPMD region to the base function....
Insert a for loop to the SPMD region....
Create a module that contains multiple functions....
Dump the MLIR AST....

module  {
  func @axpy(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: f64, %arg3: i32) {
    %c0_i32 = constant 0 : i32
    %0 = pirg.parallel_data_info (x, shared, implicit, n/a, n/a, read-only : %arg0)
    %1 = pirg.parallel_data_info (y, shared, implicit, n/a, n/a, read-write : %arg1)
    %2 = pirg.parallel_data_info (a, shared, implicit, n/a, n/a, read-only : %arg2)
    %3 = pirg.parallel_data_info (n, shared, implicit, n/a, n/a, read-only : %arg3)
    %4 = pirg.parallel_data_info (i, private, implicit, n/a, n/a, read-write : %c0_i32)
    %c6_i32 = constant 6 : i32
    pirg.spmd num_units(%c6_i32 : i32) data(%0, %1, %2, %3, %4) {
      %c0 = constant 0 : index
      %c1 = constant 1 : index
      pirg.workshare {
        scf.for %arg4 = %c0 to %arg3 step %c1 {
          %5 = memref.load %arg0[%arg4] : memref<f64>
          %6 = memref.load %arg1[%arg4] : memref<f64>
          %7 = mulf %arg2, %5 : f64
          %8 = addf %7, %6 : f64
          memref.store %8, %arg1[%arg4] : memref<f64>
        }
      }
    }
  }
}


All done....
```
