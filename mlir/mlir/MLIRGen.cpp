//===- MLIRGen.cpp - MLIR Generation from a Sage AST ----------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Sage AST
//
//===----------------------------------------------------------------------===//

#include "pirg/MLIRGen.h"
#include "pirg/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace mlir::pirg;
using namespace pirg;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {


} // namespace

namespace pirg {


} // namespace pirg
