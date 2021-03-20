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
