# UPIR-generator

UPIR generator is a tool based on REX compiler to generate UPIR, a unified parallel intermediate representation that reflects the task and data dependencies as well as other parallelism information.

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
To set up the REX compiler for building UPIR generator:

```bash
export REX_INSTALL=/opt/rex_install
export LD_LIBRARY_PATH=$REX_INSTALL/lib:$LD_LIBRARY_PATH
export PATH=$REX_INSTALL/bin:$PATH
# set up OpenJDK 1.8
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export LD_LIBRARY_PATH=$JAVA_HOME/jre/lib/amd64/server:$LD_LIBRARY_PATH
```

LLVM is also required to support MLIR. Please follow the instructions [here](https://mlir.llvm.org/getting_started) to build LLVM properly.
It could be installed to `/opt/llvm-install`.
The the opt folder would be like:
```bash
opt
├── llvm_install
│   ├── bin
│   ├── examples
│   ├── include
│   ├── lib
│   ├── libexec
│   └── share
├── rex_install
│   ├── bin
│   ├── include
│   ├── lib
│   └── share
└── ...

```

# Build

At the moment, the parallel flow graph and visualation are still work-in-progress. Please ignore them and check UPIR generator in `mlir` subfolder.

# Run

...
