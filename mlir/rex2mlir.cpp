//===----------------------------------------------------------------------===//
//
// This program:
// 1. call REX compiler to parse the input source code and generate Sage AST;
// 2. generate an MLIR AST based on the Sage AST;
// 3. dump the MLIR AST.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/MLIRGen.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"


void dummy() {

    std::cout << "Set up MLIR environment...." << std::endl;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::toy::ToyDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    mlir::OpBuilder builder = mlir::OpBuilder(&context);

    std::cout << "Prepare a dummy code location...." << std::endl;
    mlir::Location location = builder.getUnknownLoc();

    std::cout << "Prepare base function parameters...." << std::endl;
    llvm::ArrayRef<std::unique_ptr<llvm::StringRef>> args;
    llvm::SmallVector<mlir::Type, 4> arg_types(args.size(), builder.getNoneType());
    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    llvm::ArrayRef<std::pair<mlir::Identifier, mlir::Attribute> > attrs = {};

    std::cout << "Prepare base function name...." << std::endl;
    llvm::StringRef func_name = std::string("foo");

    std::cout << "Create a base function...." << std::endl;
    mlir::FuncOp func = mlir::FuncOp::create(location, func_name, func_type, attrs);

    std::cout << "Create the body of base function...." << std::endl;
    mlir::Block &entryBlock = *func.addEntryBlock();

    builder.setInsertionPointToStart(&entryBlock);

    std::cout << "Insert a SPMD region to the base function...." << std::endl;
    mlir::Value num_threads = builder.create<mlir::ConstantIntOp>(location, 6, 32);
    mlir::toy::SpmdOp spmd = builder.create<mlir::toy::SpmdOp>(location, num_threads);
    mlir::Region &spmd_body = spmd.getRegion();
    builder.createBlock(&spmd_body);

    std::cout << "Insert a for loop to the SPMD region...." << std::endl;
    mlir::Value upper_bound = builder.create<mlir::ConstantIndexOp>(location, 0);
    mlir::Value lower_bound = builder.create<mlir::ConstantIndexOp>(location, 10);
    mlir::Value step = builder.create<mlir::ConstantIndexOp>(location, 1);
    mlir::ValueRange loop_value = {};
    mlir::scf::ForOp loop = builder.create<mlir::scf::ForOp>(location, upper_bound, lower_bound, step, loop_value);
    mlir::Region &loop_body = loop.getLoopBody();
    mlir::Block &loop_block = loop_body.front();
    builder.setInsertionPointToStart(&loop_block);

    std::cout << "Insert a printf function call to the for loop...." << std::endl;
    llvm::StringRef print_name = std::string("printf");
    mlir::StringAttr print_string = builder.getStringAttr(llvm::StringRef("This is a test.\n"));
    mlir::Value print_value = builder.create<mlir::ConstantOp>(location, print_string);
    mlir::ValueRange print_value_range = mlir::ValueRange(print_value);
    mlir::TypeRange print_type = mlir::TypeRange(print_value_range);
    builder.create<mlir::CallOp>(location, print_name, print_type, print_value_range);

    std::cout << "Create a module that contains multiple functions...." << std::endl;
    mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    theModule.push_back(func);

    mlir::OwningModuleRef module = theModule;
    assert (module);

    std::cout << "Dump the MLIR AST...." << std::endl;
    module->dump();
    std::cout << "All done...." << std::endl;
}


int main(int argc, char **argv) {
  // Register required dialects
  mlir::DialectRegistry registry;
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  dummy();

  return 0;
}
