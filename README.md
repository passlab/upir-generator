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

For now, it will generate the REX AST but use a dummy task graph for visualization.

```bash
./pfg.out test.c
```
