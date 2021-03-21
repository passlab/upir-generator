# parallel-flow-graph-generator

PFGG is a tool based on REX compiler to generate a parallel flow graph, which reflects the task and data dependencies as well as other parallelism information.

# Prerequisite

REX compiler is required. Any other dependencies will be installed while building REX. Please check this [guide](https://github.com/passlab/rexompiler/wiki/REX-compiler-compilation) for REX installation.
Then the environment variable `REX_INSTALL` must be properly set to indicates the location of REX compiler.

For example, if the REX compiler is installed under `/opt/rex_install` as follows,
```bash
opt
├── rex_install
│   ├── bin
│   ├── include
│   ├── lib
│   └── share
└── ...

```
To set up the REX compiler for building PFGG:

```bash
export REX_INSTALL=/opt/rex_install
export LD_LIBRARY_PATH=$REX_INSTALL/lib:$LD_LIBRARY_PATH
export PATH=$REX_INSTALL/bin:$PATH
# set up OpenJDK 1.8
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export LD_LIBRARY_PATH=$JAVA_HOME/jre/lib/amd64/server:$LD_LIBRARY_PATH
```


# Build

```bash
make
```

# Run

It will generate a task graph based on the REX AST, but for now, the visualization is only to list all the task nodes in the graph in pre-order.
Please notice that the flag `-rose:openmp:ast_only` has to be specified.

```bash
./pfgg.out -rose:openmp:ast_only test.c
```

For the given input `test.c`, the SgStatements in the task graph would be:

```bash
Check the task graph....
SgNode: SgFunctionDefinition at line: 5
SgNode: SgVariableDeclaration at line: 7
SgNode: SgOmpParallelStatement at line: 10
SgNode: SgVariableDeclaration at line: 12
SgNode: SgOmpForStatement at line: 15
SgNode: SgForStatement at line: 16
SgNode: SgExprStatement at line: 17
SgNode: SgExprStatement at line: 19
SgNode: SgOmpBarrierStatement at line: 20
SgNode: SgExprStatement at line: 21
SgNode: SgExprStatement at line: 24
```
