//===- Dialect.cpp - Pirg IR Dialect registration in MLIR -----------------===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements the Pirg dialect
//
//===----------------------------------------------------------------------===//

#include "pirg/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::pirg;

//===----------------------------------------------------------------------===//
// PirgDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void PirgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pirg/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Pirg Operations
//===----------------------------------------------------------------------===//


static void printSpmdOp(mlir::OpAsmPrinter &printer, mlir::pirg::SpmdOp op) {
  printer << "pirg.spmd";
  if (auto threads = op.num_units())
    printer << " num_units(" << threads << " : " << threads.getType() << ")";

  printer.printRegion(op.getRegion());
}


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "pirg/Ops.cpp.inc"
