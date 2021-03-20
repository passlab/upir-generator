# openmp-graph-generator
A tool based on REX compiler to generate OpenMP task and data graph.

# Prerequisite

REX compiler is required. Please check this [guide](https://github.com/passlab/rexompiler/wiki/REX-compiler-compilation) for installation.
Then the environment variables `REX_ROOT` and `REX_INSTALL` must be properly set to indicates the location of REX compiler.


# Build

```bash
make
```

# Run

It will generate a task graph based on the REX AST, but for now, the visualization is only to list all the task nodes in the graph in pre-order.

```bash
./pfg.out -rose:openmp:ast_only test.c
```

For the given input `test.c`, the SgStatements in the task graph would be:

```bash
Check the task graph....
SgNode: SgFunctionDefinition at line: 5
SgNode: SgVariableDeclaration at line: 7
SgNode: SgPragmaDeclaration at line: 10
SgNode: SgVariableDeclaration at line: 12
SgNode: SgPragmaDeclaration at line: 15
SgNode: SgForStatement at line: 16
SgNode: SgExprStatement at line: 19
SgNode: SgPragmaDeclaration at line: 20
SgNode: SgExprStatement at line: 21
SgNode: SgExprStatement at line: 24
```
