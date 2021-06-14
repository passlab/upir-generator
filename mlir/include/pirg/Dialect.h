//===- Dialect.h - Dialect definition for the Pirg IR ----------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR of Pirg Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the
/// pirg dialect.
#include "pirg/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// pirg operations.
#define GET_OP_CLASSES
#include "pirg/Ops.h.inc"

