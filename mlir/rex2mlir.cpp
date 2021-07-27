//===----------------------------------------------------------------------===//
//
// This program:
// 1. call REX compiler to parse the input source code and generate Sage AST;
// 2. generate an MLIR AST based on the Sage AST;
// 3. dump the MLIR AST.
//
//===----------------------------------------------------------------------===//

#include "pirg/Dialect.h"
#include "pirg/MLIRGen.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "llvm/ADT/ScopedHashTable.h"

//#include "rose.h"
#include "data_analyzing.h"

void dummy() {

    std::cout << "Set up MLIR environment...." << std::endl;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::pirg::PirgDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
    context.getOrLoadDialect<mlir::acc::OpenACCDialect>();
    mlir::OpBuilder builder = mlir::OpBuilder(&context);

    std::cout << "Prepare a dummy code location...." << std::endl;
    mlir::Location location = builder.getUnknownLoc();

    std::cout << "Prepare base function parameters...." << std::endl;

    // func paras
    llvm::SmallVector<mlir::Type, 4> arg_types;
    llvm::ArrayRef<int64_t> foo_array;
    mlir::MemRefType foo_xy = mlir::MemRefType::Builder(foo_array, builder.getF64Type());
    arg_types.push_back(foo_xy);
    arg_types.push_back(foo_xy);
    mlir::FloatType foo_a = builder.getF64Type();
    arg_types.push_back(foo_a);
    mlir::IntegerType foo_n = builder.getI32Type();
    arg_types.push_back(foo_n);

    auto func_type = builder.getFunctionType(arg_types, llvm::None);

    llvm::StringRef func_name = std::string("axpy");

    std::cout << "Create a base function...." << std::endl;
    mlir::FuncOp func = builder.create<mlir::FuncOp>(location, func_name, func_type);

    std::cout << "Create the body of base function...." << std::endl;
    mlir::Block &entryBlock = *func.addEntryBlock();

    builder.setInsertionPointToStart(&entryBlock);

    mlir::Value var_i = builder.create<mlir::ConstantIntOp>(location, 0, 32);
    // parallel data
    // x
    llvm::ArrayRef<llvm::StringRef> parallel_data_string_x = {"x", "shared", "implicit", "", "", "read-only"};
    mlir::ArrayAttr parallel_data_x = builder.getStrArrayAttr(parallel_data_string_x);
    mlir::Value parallel_data_value_x = builder.create<mlir::pirg::ParallelDataInfoOp>(location, builder.getNoneType(), parallel_data_x, entryBlock.getArgument(0));
    // y
    llvm::ArrayRef<llvm::StringRef> parallel_data_string_y = {"y", "shared", "implicit", "", "", "read-write"};
    mlir::ArrayAttr parallel_data_y = builder.getStrArrayAttr(parallel_data_string_y);
    mlir::Value parallel_data_value_y = builder.create<mlir::pirg::ParallelDataInfoOp>(location, builder.getNoneType(), parallel_data_y, entryBlock.getArgument(1));
    // a
    llvm::ArrayRef<llvm::StringRef> parallel_data_string_a = {"a", "shared", "implicit", "", "", "read-only"};
    mlir::ArrayAttr parallel_data_a = builder.getStrArrayAttr(parallel_data_string_a);
    mlir::Value parallel_data_value_a = builder.create<mlir::pirg::ParallelDataInfoOp>(location, builder.getNoneType(), parallel_data_a, entryBlock.getArgument(2));
    // n
    llvm::ArrayRef<llvm::StringRef> parallel_data_string_n = {"n", "shared", "implicit", "", "", "read-only"};
    mlir::ArrayAttr parallel_data_n = builder.getStrArrayAttr(parallel_data_string_n);
    mlir::Value parallel_data_value_n = builder.create<mlir::pirg::ParallelDataInfoOp>(location, builder.getNoneType(), parallel_data_n, entryBlock.getArgument(3));
    // i
    llvm::ArrayRef<llvm::StringRef> parallel_data_string_i = {"i", "private", "implicit", "", "", "read-write"};
    mlir::ArrayAttr parallel_data_i = builder.getStrArrayAttr(parallel_data_string_i);
    mlir::Value parallel_data_value_i = builder.create<mlir::pirg::ParallelDataInfoOp>(location, builder.getNoneType(), parallel_data_i, var_i);

    llvm::ArrayRef<mlir::Value> parallel_data_values = {parallel_data_value_x, parallel_data_value_y, parallel_data_value_a, parallel_data_value_n, parallel_data_value_i};
    mlir::ValueRange parallel_data_range = mlir::ValueRange(parallel_data_values);

    std::cout << "Insert a SPMD region to the base function...." << std::endl;
    mlir::Value num_units = builder.create<mlir::ConstantIntOp>(location, 6, 32);
    mlir::pirg::SpmdOp spmd = builder.create<mlir::pirg::SpmdOp>(location, num_units, nullptr, mlir::ValueRange(), parallel_data_range, nullptr);
    mlir::Region &spmd_body = spmd.getRegion();
    builder.createBlock(&spmd_body);

    std::cout << "Insert a for loop to the SPMD region...." << std::endl;
    mlir::Value lower_bound = builder.create<mlir::ConstantIndexOp>(location, 0);
    mlir::Value upper_bound = entryBlock.getArgument(3);
    mlir::Value step = builder.create<mlir::ConstantIndexOp>(location, 1);

    mlir::pirg::WorkshareOp workshare_target = builder.create<mlir::pirg::WorkshareOp>(location, nullptr, lower_bound, upper_bound, step, nullptr, nullptr, nullptr, mlir::ValueRange(), nullptr, mlir::ValueRange(), nullptr);
    mlir::Region &workshare_body = workshare_target.getRegion();
    builder.createBlock(&workshare_body);

    mlir::ValueRange loop_value = {};
    mlir::scf::ForOp loop = builder.create<mlir::scf::ForOp>(location, lower_bound, upper_bound, step, loop_value);
    mlir::Region &loop_body = loop.getLoopBody();
    mlir::Block &loop_block = loop_body.front();
    builder.setInsertionPointToStart(&loop_block);

    mlir::Value loop_index = loop.getInductionVar();
    mlir::Value foo_x_val = entryBlock.getArgument(0); // x
    mlir::Value foo_y_val = entryBlock.getArgument(1); // y
    mlir::Value foo_a_val = entryBlock.getArgument(2);
    mlir::memref::LoadOp foo_x_i = builder.create<mlir::memref::LoadOp>(location, foo_x_val, loop_index);
    mlir::memref::LoadOp foo_y_i = builder.create<mlir::memref::LoadOp>(location, foo_y_val, loop_index);

    mlir::MulFOp a_mul_x = builder.create<mlir::MulFOp>(location, foo_a_val, foo_x_i);
    mlir::AddFOp new_y = builder.create<mlir::AddFOp>(location, a_mul_x, foo_y_i);

    builder.create<mlir::memref::StoreOp>(location, new_y, foo_y_val, loop_index);

    std::cout << "Create a module that contains multiple functions...." << std::endl;
    mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    theModule.push_back(func);

    mlir::OwningModuleRef module = theModule;
    assert (module);

    std::cout << "Dump the MLIR AST....\n" << std::endl;
    module->dump();

    /*
    std::cout << "\nConvert Pirg dialect to OpenMP dialect....\n" << std::endl;
    mlir::IRRewriter rewriter = mlir::IRRewriter(&context);

    mlir::Value omp_num_threads = spmd.num_units();
    rewriter.setInsertionPointAfter(spmd);
    mlir::omp::ParallelOp omp_parallel = rewriter.create<mlir::omp::ParallelOp>(location, nullptr, omp_num_threads, nullptr, mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), nullptr);
    rewriter.inlineRegionBefore(spmd.region(), omp_parallel.region(), omp_parallel.region().begin());
    rewriter.eraseOp(spmd);
    */

    std::cout << "\nConvert Pirg dialect to OpenACC dialect....\n" << std::endl;
    mlir::IRRewriter rewriter = mlir::IRRewriter(&context);

    mlir::Value acc_num_workers = spmd.num_units();
    rewriter.setInsertionPointAfter(spmd);
    mlir::acc::ParallelOp acc_parallel = rewriter.create<mlir::acc::ParallelOp>(location, nullptr, nullptr, mlir::ValueRange(), nullptr, nullptr, acc_num_workers, nullptr, nullptr, nullptr, nullptr, nullptr, mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), mlir::ValueRange(), nullptr);
    rewriter.inlineRegionBefore(spmd.region(), acc_parallel.region(), acc_parallel.region().begin());
    rewriter.eraseOp(spmd);

    rewriter.setInsertionPointAfter(workshare_target);
    mlir::acc::LoopOp acc_loop = rewriter.create<mlir::acc::LoopOp>(location, mlir::TypeRange(), nullptr, nullptr, nullptr, nullptr, nullptr, false, false, false, mlir::ValueRange(), mlir::ValueRange(), nullptr, mlir::ValueRange(), mlir::acc::OpenACCExecMapping::WORKER);
    rewriter.inlineRegionBefore(workshare_target.region(), acc_loop.region(), acc_loop.region().begin());
    rewriter.eraseOp(workshare_target);

    rewriter.eraseOp(parallel_data_value_x.getDefiningOp());
    rewriter.eraseOp(parallel_data_value_y.getDefiningOp());
    rewriter.eraseOp(parallel_data_value_a.getDefiningOp());
    rewriter.eraseOp(parallel_data_value_n.getDefiningOp());
    rewriter.eraseOp(parallel_data_value_i.getDefiningOp());
    module->dump();

    std::cout << "\nAll done...." << std::endl;
}


int main(int argc, char **argv) {
  // Register required dialects
  mlir::DialectRegistry registry;
  registry.insert<mlir::StandardOpsDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::omp::OpenMPDialect>();
  registry.insert<mlir::acc::OpenACCDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  dummy();

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::pirg::PirgDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  SgProject* project = frontend(argc, argv);
  assert(project);
  std::map<SgVariableSymbol *, ParallelData *> parallel_data = analyze_parallel_data(isSgSourceFile(&(project->get_file(0))));
  std::map<SgVariableSymbol *, ParallelData *>::iterator iter;
  for (iter = parallel_data.begin(); iter != parallel_data.end(); iter++) {
    iter->second->output();
  }

  mlir::OwningModuleRef mlir_module = generate_mlir(context, project);
  mlir_module->dump();

  return 0;
}
