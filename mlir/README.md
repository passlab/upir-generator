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
void foo () {
#pragma omp parallel num_threads(6)
    for (i = 0; i < 10; i++) {
        printf("This is a test.\n");
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
Prepare base function name....
Create a base function....
Create the body of base function....
Insert a SPMD region to the base function....
Insert a for loop to the SPMD region....
Insert a printf function call to the for loop....
Create a module that contains multiple functions....
Dump the MLIR AST....
module  {
  func @foo() {
    %c6_i32 = constant 6 : i32
    parallel.spmd num_threads(%c6_i32 : i32) {
      %c0 = constant 0 : index
      %c10 = constant 10 : index
      %c1 = constant 1 : index
      scf.for %arg0 = %c0 to %c10 step %c1 {
        %cst = constant "This is a test.\0A"
        %0 = call @printf(%cst) : (none) -> none
      }
    }
  }
}
All done....
```
