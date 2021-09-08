//===- MLIRGen.h - MLIR Generation from a Upir AST -------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include <memory>

#include <vector>
#include <string>
#include "rose.h"

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace upir {

} // namespace upir

class InheritedAttribute {

    public:
        SgNode* frontier = NULL;
        bool to_explore = false;
        size_t depth = 0;
        SgNode* expr_parent = NULL;

        InheritedAttribute(SgNode* __frontier, bool __to_explore, size_t __depth, SgNode* __expr_parent) : frontier(__frontier), to_explore(__to_explore), depth(__depth), expr_parent(__expr_parent) { };

};


mlir::ModuleOp generate_mlir(mlir::MLIRContext&, SgProject*);

void convert_basic_block(mlir::OpBuilder&, SgBasicBlock*);

void convert_statement(mlir::OpBuilder&, SgStatement*);
mlir::Value convert_binary_op(mlir::OpBuilder&, SgExpression*);
mlir::Value convert_op(mlir::OpBuilder&, SgExpression*);
