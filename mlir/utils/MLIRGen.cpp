//===- MLIRGen.cpp - MLIR Generation from a Sage AST ----------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Sage AST
//
//===----------------------------------------------------------------------===//

#include "upir/MLIRGen.h"
#include "upir/Dialect.h"

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

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "mlir/IR/OwningOpRef.h"

using namespace mlir::upir;

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

namespace upir {

std::map<SgNode*, std::pair<mlir::Value, SgInitializedName*>> symbol_table;
SgProject* g_project = NULL;

} // namespace upir


SgInitializedName* get_sage_symbol(SgExpression* node) {

    std::cout << "======  Checking a SgExpression symbol: " << node->sage_class_name() << " ======\n";
    SgInitializedName* symbol = NULL;
    SgVarRefExp* symbol_expression = isSgVarRefExp(node);
    if (symbol_expression != NULL) {
        symbol = symbol_expression->get_symbol()->get_declaration();
    }

    return symbol;
}

mlir::Value convert_binary_op(mlir::OpBuilder& builder, SgExpression* node) {

    mlir::Location location = builder.getUnknownLoc();
    mlir::Value result = nullptr;
    mlir::LLVM::FMFAttr fmf = mlir::LLVM::FMFAttr::get(builder.getContext(), {});
    // TODO: implement common binary ops
    switch (node->variantT()) {
        case V_SgAddOp:
            {
                SgAddOp* op = isSgAddOp(node);
                assert(op != NULL);
                mlir::Value left_operand = convert_op(builder, op->get_lhs_operand());
                mlir::Value right_operand = convert_op(builder, op->get_rhs_operand());
                result = builder.create<mlir::LLVM::FAddOp>(location, left_operand, right_operand, fmf);
                break;
            }
        case V_SgMultiplyOp:
            {
                SgMultiplyOp* op = isSgMultiplyOp(node);
                assert(op != NULL);
                mlir::Value left_operand = convert_op(builder, op->get_lhs_operand());
                mlir::Value right_operand = convert_op(builder, op->get_rhs_operand());
                result = builder.create<mlir::LLVM::FMulOp>(location, left_operand, right_operand, fmf);
                break;
            }
        case V_SgDivideOp:
            {
                SgDivideOp* op = isSgDivideOp(node);
                assert(op != NULL);
                mlir::Value left_operand = convert_op(builder, isSgBinaryOp(node)->get_lhs_operand());
                mlir::Value right_operand = convert_op(builder, isSgBinaryOp(node)->get_rhs_operand());
                result = builder.create<mlir::LLVM::FDivOp>(location, left_operand, right_operand, fmf);
                break;
            }
        case V_SgAssignOp:
            {
                SgAssignOp* assign_op = isSgAssignOp(node);
                assert(assign_op != NULL);
                try {
                    std::stoi(assign_op->get_rhs_operand()->unparseToString());
                }
                catch (...) {
                    break;
                }

                SgInitializedName *init_var = get_sage_symbol(assign_op->get_lhs_operand());
                ROSE_ASSERT(init_var != NULL);
                SgVariableSymbol *symbol = isSgVariableSymbol(init_var->search_for_symbol_from_symbol_table());
                assert(symbol != NULL);
                assert(upir::symbol_table.count(init_var) != 0);
                mlir::Value value = convert_op(builder, assign_op->get_rhs_operand());
                upir::symbol_table[init_var] = std::make_pair(value, init_var);
                break;
            }
        default:
            {
                printf("Unknown Binary SgExpression!\n");
            }
    }
    return result;
}

mlir::Value convert_op(mlir::OpBuilder& builder, SgExpression* node) {

    mlir::Location location = builder.getUnknownLoc();
    mlir::Value result = nullptr;
    //SgGlobal* global_scope = SageInterface::getGlobalScope(node);
    switch (node->variantT()) {
        /*
        case V_SgCudaKernelCallExp:
            {
                SgCudaKernelCallExp* cuda_kernel = isSgCudaKernelCallExp(node);
                assert(cuda_kernel != NULL);
                // due to the implementation of REX/ROSE, only 1D kernel is supported.
                // <<<number of blocks, number of threads per block>>>
                SgCudaKernelExecConfig* kernel_config = cuda_kernel->get_exec_config();

                mlir::Value num_blocks = nullptr;
                SgExpression* num_blocks_expression = kernel_config->get_grid();
                num_blocks = convert_op(builder, num_blocks_expression);

                mlir::Value num_threads_per_block = nullptr;
                SgExpression* num_threads_per_block_expression = kernel_config->get_blocks();
                num_threads_per_block = convert_op(builder, num_threads_per_block_expression);

                SgExpressionPtrList& func_parameters = cuda_kernel->get_args()->get_expressions();
                std::map<SgVariableSymbol *, ParallelData *> parallel_data = analyze_cuda_parallel_data(cuda_kernel);
                std::map<SgVariableSymbol *, ParallelData *>::iterator data_iter;
                std::vector<mlir::Value> value_list;
                int i = 0;
                for (data_iter = parallel_data.begin(); data_iter != parallel_data.end(); data_iter++) {
                  ParallelData * iter = data_iter->second;
                  std::string symbol = iter->get_symbol()->get_name();
                  std::string sharing_property = iter->get_sharing_property();
                  std::string sharing_visibility = iter->get_sharing_visibility();
                  std::string mapping_property = iter->get_mapping_property();
                  std::string mapping_visibility = iter->get_mapping_visibility();
                  std::string data_access = iter->get_data_access();
                  llvm::ArrayRef<llvm::StringRef> parallel_data_string = {symbol, sharing_property, sharing_visibility, mapping_property, mapping_visibility, data_access};
                  mlir::ArrayAttr parallel_data = builder.getStrArrayAttr(parallel_data_string);
                  SgInitializedName *sage_symbol = get_sage_symbol(func_parameters.at(i));
                  i++;
                  mlir::Value symbol_value = nullptr;
                  if (sage_symbol != NULL) {
                      symbol_value = upir::symbol_table[sage_symbol].first;
                  }
                  mlir::Value parallel_data_value = builder.create<mlir::upir::ParallelDataInfoOp>(location, builder.getNoneType(), parallel_data, symbol_value);
                  value_list.push_back(parallel_data_value);

                }
                llvm::ArrayRef<mlir::Value> parallel_data_values = llvm::ArrayRef<mlir::Value>(value_list);
                mlir::ValueRange parallel_data_range = mlir::ValueRange(parallel_data_values);

                mlir::StringAttr device = builder.getStringAttr(llvm::StringRef("nvptx"));
                mlir::upir::TaskOp task_target = builder.create<mlir::upir::TaskOp>(location, nullptr, device, nullptr, mlir::ValueRange(), parallel_data_range, nullptr);
                mlir::Region &task_body = task_target.getRegion();
                builder.createBlock(&task_body);

                mlir::upir::SpmdOp spmd_grid = builder.create<mlir::upir::SpmdOp>(location, num_blocks, nullptr, mlir::ValueRange(), parallel_data_range, nullptr);
                mlir::Region &spmd_grid_body = spmd_grid.getRegion();
                builder.createBlock(&spmd_grid_body);

                mlir::upir::SpmdOp spmd_block = builder.create<mlir::upir::SpmdOp>(location, num_threads_per_block, nullptr, mlir::ValueRange(), parallel_data_range, nullptr);
                mlir::Region &spmd_block_body = spmd_block.getRegion();
                builder.createBlock(&spmd_block_body);

                SgExpression* func_name = cuda_kernel->get_function();
                std::string func_name_string = func_name->unparseToString();
                SgFunctionSymbol* func_symbol = global_scope->lookup_function_symbol(func_name_string);
                assert(func_symbol != NULL);

                std::vector<mlir::Value> args;
                SgExpressionPtrList::const_iterator iter;
                for (iter = func_parameters.begin(); iter != func_parameters.end(); iter++) {
                    mlir::Value arg = nullptr;
                    SgInitializedName* symbol = get_sage_symbol(*iter);
                    assert(symbol != NULL);
                    arg = upir::symbol_table.at(symbol).first;
                    args.push_back(arg);
                }
                llvm::ArrayRef<mlir::Value> args_value = llvm::ArrayRef<mlir::Value>(args);
                mlir::ValueRange func_parameter_values = mlir::ValueRange(args_value);
                builder.create<mlir::CallOp>(location, llvm::StringRef(func_name_string), mlir::TypeRange(), func_parameter_values);

                builder.setInsertionPointAfter(task_target);
                break;
            }
    */
        case V_SgAddOp:
        case V_SgAssignOp:
        case V_SgDivideOp:
        case V_SgMultiplyOp:
            {
                SgBinaryOp* target = isSgBinaryOp(node);
                assert(target != NULL);
                result = convert_binary_op(builder, target);
                break;
            }
        case V_SgVarRefExp:
            {
                SgInitializedName* symbol = get_sage_symbol(node);
                result = upir::symbol_table.at(symbol).first;
                assert(result != nullptr);
                break;
            }
        case V_SgIntVal:
            {
                result = builder.create<mlir::arith::ConstantIntOp>(location, std::stoi(node->unparseToString()), 32);
                break;
            }
        default:
            {
                printf("Unknown SgExpression!\n");
                std::cout << "======  Meet a new SgExpression Op: " << node->sage_class_name() << " ======\n";
            }
    }
    return result;
}

void convert_statement(mlir::OpBuilder& builder, SgStatement* node) {

    if (node->get_file_info()->get_line() == 0) {
        return;
    }
    mlir::Location location = builder.getUnknownLoc();
    switch (node->variantT()) {
        case V_SgVariableDeclaration:
            {
                SgVariableDeclaration* target = isSgVariableDeclaration(node);
                SgInitializedNamePtrList variable_list = target->get_variables();
                SgInitializedNamePtrList::const_iterator iter;
                for (iter = variable_list.begin(); iter != variable_list.end(); iter++) {
                    mlir::Type type = nullptr;
                    SgInitializedName* symbol = *iter;
                    SgType* symbol_type = symbol->get_type();
                    if (isSgPointerType(symbol_type)) {
                        type = mlir::UnrankedMemRefType::get(builder.getI32Type(), 8);
                    } else {
                        type = builder.getI32Type();
                    }
                    assert(upir::symbol_table.count(symbol) == 0);
                    upir::symbol_table[symbol] = std::make_pair(nullptr, symbol);
                }
                break;
            }
        case V_SgExprStatement:
            {
                SgExprStatement* target = isSgExprStatement(node);
                assert(target != NULL);
                convert_op(builder, target->get_expression());
                break;
            }
        case V_SgUpirSpmdStatement:
            {
                SgUpirSpmdStatement* target = isSgUpirSpmdStatement(node);
                std::cout << "Insert a SPMD region...." << std::endl;

                //std::map<SgVariableSymbol *, ParallelData *> parallel_data = analyze_parallel_data(isSgSourceFile(&(upir::g_project->get_file(0))));
                //std::map<SgVariableSymbol *, ParallelData *>::iterator data_iter;
                Rose_STL_Container<SgOmpClause *> data_fields = OmpSupport::getClause(target, V_SgUpirDataField);
                std::vector<mlir::Value> value_list;
                for (size_t i = 0; i < data_fields.size(); i++) {
                  //ParallelData * iter = data_iter->second;
                  SgUpirDataField* data_field = (SgUpirDataField*)data_fields[i];
                  ROSE_ASSERT(data_field);
                  std::list<SgUpirDataItemField *> data_items = data_field->get_data();
                  for (std::list<SgUpirDataItemField*>::iterator iter = data_items.begin(); iter != data_items.end(); iter++) {
                      SgUpirDataItemField* data_item = *iter;
                      std::string symbol = data_item->get_symbol()->get_name();
                  std::string sharing_property = "";
                  std::string sharing_visibility = "";
                  std::string mapping_property = "";
                  std::string mapping_visibility = "";
                  std::string data_access = "";
                  llvm::ArrayRef<llvm::StringRef> parallel_data_string = {symbol, sharing_property, sharing_visibility, mapping_property, mapping_visibility, data_access};
                  mlir::ArrayAttr parallel_data = builder.getStrArrayAttr(parallel_data_string);
                  mlir::Value parallel_data_value = builder.create<mlir::upir::ParallelDataInfoOp>(location, builder.getNoneType(), parallel_data, nullptr);
                  value_list.push_back(parallel_data_value);
                  }

                }
                llvm::ArrayRef<mlir::Value> parallel_data_values = llvm::ArrayRef<mlir::Value>(value_list);
                mlir::ValueRange parallel_data_range = mlir::ValueRange(parallel_data_values);

                mlir::Value num_threads = nullptr;
                if (OmpSupport::hasClause(target, V_SgUpirNumUnitsField)) {
                    Rose_STL_Container<SgOmpClause*> num_threads_clauses = OmpSupport::getClause(target, V_SgUpirNumUnitsField);
                    SgUpirNumUnitsField* num_threads_clause = isSgUpirNumUnitsField(num_threads_clauses[0]);
                    std::string num_threads_string = num_threads_clause->get_expression()->unparseToString();
                    int32_t num_thread_value = std::stoi(num_threads_string);
                    num_threads = builder.create<mlir::arith::ConstantIntOp>(location, num_thread_value, 32);
                }
                mlir::upir::SpmdOp spmd = builder.create<mlir::upir::SpmdOp>(location, num_threads, nullptr, mlir::ValueRange(), parallel_data_range, nullptr);
                mlir::Region &spmd_body = spmd.getRegion();
                builder.createBlock(&spmd_body);

                SgStatement* omp_parallel_body = target->get_body();
                if (isSgBasicBlock(omp_parallel_body)) {
                    convert_basic_block(builder, isSgBasicBlock(omp_parallel_body));
                } else {
                    convert_statement(builder, omp_parallel_body);
                }
                builder.setInsertionPointAfter(spmd);
                break;
            }
        case V_SgUpirLoopParallelStatement:
        case V_SgOmpDoStatement:
            {
                SgUpirLoopParallelStatement* omp_for = isSgUpirLoopParallelStatement(node);
                SgOmpDoStatement* omp_do = isSgOmpDoStatement(node);
                SgUpirFieldBodyStatement* omp_target = NULL;
                if (omp_for != NULL) {
                    omp_target = isSgUpirLoopStatement(omp_for->get_loop());
                } else {
                    omp_target = omp_do;
                }
                assert(omp_target != NULL);
                SgForStatement* for_loop = isSgForStatement(omp_target->get_body());
                SgFortranDo* do_loop = isSgFortranDo(omp_target->get_body());
                std::cout << "Insert a worksharing region...." << std::endl;

                mlir::Value collapse = nullptr;
                if (OmpSupport::hasClause(omp_target, V_SgOmpCollapseClause)) {
                    Rose_STL_Container<SgOmpClause*> collapse_clauses = OmpSupport::getClause(omp_target, V_SgOmpCollapseClause);
                    SgOmpCollapseClause* collapse_clause = isSgOmpCollapseClause(collapse_clauses[0]);
                    std::string collapse_string = collapse_clause->get_expression()->unparseToString();
                    int32_t collapse_value = std::stoi(collapse_string);
                    collapse = builder.create<mlir::arith::ConstantIntOp>(location, collapse_value, 32);
                }
                SgInitializedName* for_index = NULL;
                SgExpression* for_lower = NULL;
                SgExpression* for_upper = NULL;
                SgExpression* for_stride = NULL;
                bool isIncremental = true;
                bool is_canonical = false;

                if (for_loop != NULL) {
                    is_canonical = SageInterface::isCanonicalForLoop(for_loop, &for_index, &for_lower, &for_upper, &for_stride, NULL, &isIncremental);
                } else {
                    SageInterface::doLoopNormalization(do_loop);
                    is_canonical = SageInterface::isCanonicalDoLoop(do_loop, &for_index, &for_lower, &for_upper, &for_stride, NULL, &isIncremental, NULL);
                }
                ROSE_ASSERT(is_canonical == true);

                mlir::Value lower_bound = nullptr;
                SgInitializedName* for_lower_symbol = get_sage_symbol(for_lower);
                if (for_lower_symbol != NULL) {
                    lower_bound = upir::symbol_table.at(for_lower_symbol).first;
                } else {
                    if (upir::symbol_table.count(for_lower) != 0) {
                        lower_bound = upir::symbol_table[for_lower].first;
                    } else {
                        lower_bound = builder.create<mlir::arith::ConstantIndexOp>(location, std::stoi(for_lower->unparseToString()));
                        upir::symbol_table[for_lower] = std::make_pair(lower_bound, for_lower_symbol);
                    }
                }
                mlir::Value upper_bound = nullptr;
                SgInitializedName* for_upper_symbol = get_sage_symbol(for_upper);
                if (for_upper_symbol != NULL) {
                    upper_bound = upir::symbol_table.at(for_upper_symbol).first;
                    assert(upper_bound != nullptr);
                } else {
                    upper_bound = builder.create<mlir::arith::ConstantIndexOp>(location, std::stoi(for_upper->unparseToString()));
                }

                mlir::Value stride = nullptr;
                SgInitializedName* for_stride_symbol = get_sage_symbol(for_stride);
                if (for_stride_symbol != NULL) {
                    stride = upir::symbol_table.at(for_stride_symbol).first;
                } else {
                    if (upir::symbol_table.count(for_stride_symbol) != 0) {
                        stride = upir::symbol_table.at(for_stride).first;
                    } else {
                        stride = builder.create<mlir::arith::ConstantIndexOp>(location, std::stoi(for_stride->unparseToString()));
                        upir::symbol_table[for_stride] = std::make_pair(stride, for_stride_symbol);
                    }
                }

                mlir::upir::WorkshareOp workshare_target = builder.create<mlir::upir::WorkshareOp>(location, nullptr, lower_bound, upper_bound, stride, nullptr, nullptr, collapse, mlir::ValueRange(), nullptr, mlir::ValueRange(), nullptr);
                mlir::Region &workshare_body = workshare_target.getRegion();
                builder.createBlock(&workshare_body);

                SgStatement* omp_workshare_body = omp_target->get_body();
                if (isSgBasicBlock(omp_workshare_body)) {
                    convert_basic_block(builder, isSgBasicBlock(omp_workshare_body));
                } else {
                    convert_statement(builder, omp_workshare_body);
                }

                builder.setInsertionPointAfter(workshare_target);
                break;
            }
        case V_SgUpirTaskStatement:
            {
                SgUpirTaskStatement* target = isSgUpirTaskStatement(node);
                std::cout << "Insert a target region...." << std::endl;
                mlir::StringAttr device = builder.getStringAttr(llvm::StringRef("nvptx"));
                mlir::upir::TaskOp task_target = builder.create<mlir::upir::TaskOp>(location, nullptr, device, nullptr, mlir::ValueRange(), mlir::ValueRange(), nullptr);
                mlir::Region &task_body = task_target.getRegion();
                builder.createBlock(&task_body);

                SgStatement* omp_target_body = target->get_body();
                if (isSgBasicBlock(omp_target_body)) {
                    convert_basic_block(builder, isSgBasicBlock(omp_target_body));
                } else {
                    convert_statement(builder, omp_target_body);
                }
                builder.setInsertionPointAfter(task_target);
                break;
            }
        case V_SgOmpTaskStatement:
            {
                SgOmpTaskStatement* target = isSgOmpTaskStatement(node);
                std::cout << "Insert a task region...." << std::endl;
                mlir::upir::TaskOp task = builder.create<mlir::upir::TaskOp>(location, nullptr, nullptr, nullptr, mlir::ValueRange(), mlir::ValueRange(), nullptr);
                mlir::Region &task_body = task.getRegion();
                builder.createBlock(&task_body);

                SgStatement* omp_task_body = target->get_body();
                if (isSgBasicBlock(omp_task_body)) {
                    convert_basic_block(builder, isSgBasicBlock(omp_task_body));
                } else {
                    convert_statement(builder, omp_task_body);
                }
                builder.setInsertionPointAfter(task);
                break;
            }
        case V_SgForStatement:
        case V_SgFortranDo:
            {
                SgForStatement* for_loop = isSgForStatement(node);
                SgFortranDo* do_loop = isSgFortranDo(node);
                SgStatement* for_body = NULL;
                if (for_loop != NULL) {
                    for_body = for_loop->get_loop_body();
                } else {
                    for_body = do_loop->get_body();
                }
                std::cout << "Insert a for loop...." << std::endl;

                SgInitializedName* for_index = NULL;
                SgExpression* for_lower = NULL;
                SgExpression* for_upper = NULL;
                SgExpression* for_stride = NULL;
                bool isIncremental = true;
                bool is_canonical = false;
                if (for_loop != NULL) {
                    is_canonical = SageInterface::isCanonicalForLoop(for_loop, &for_index, &for_lower, &for_upper, &for_stride, NULL, &isIncremental);
                } else {
                    SageInterface::doLoopNormalization(do_loop);
                    is_canonical = SageInterface::isCanonicalDoLoop(do_loop, &for_index, &for_lower, &for_upper, &for_stride, NULL, &isIncremental, NULL);
                }
                ROSE_ASSERT(is_canonical == true);

                mlir::Value lower_bound = nullptr;
                SgInitializedName* for_lower_symbol = get_sage_symbol(for_lower);
                if (for_lower_symbol != NULL) {
                    lower_bound = upir::symbol_table.at(for_lower_symbol).first;
                } else {
                    if (upir::symbol_table.count(for_lower) != 0) {
                        lower_bound = upir::symbol_table.at(for_lower).first;
                    } else {
                        lower_bound = builder.create<mlir::arith::ConstantIndexOp>(location, std::stoi(for_lower->unparseToString()));
                        upir::symbol_table[for_lower] = std::make_pair(lower_bound, for_lower_symbol);
                    }
                }

                mlir::Value upper_bound = nullptr;
                SgInitializedName* for_upper_symbol = get_sage_symbol(for_upper);
                if (for_upper_symbol != NULL) {
                    upper_bound = upir::symbol_table.at(for_upper_symbol).first;
                } else {
                    upper_bound = builder.create<mlir::arith::ConstantIndexOp>(location, std::stoi(for_upper->unparseToString()));
                }

                mlir::Value stride = nullptr;
                SgInitializedName* for_stride_symbol = get_sage_symbol(for_stride);
                if (for_stride_symbol != NULL) {
                    stride = upir::symbol_table.at(for_stride_symbol).first;
                } else {
                    if (upir::symbol_table.count(for_stride_symbol) != 0) {
                        stride = upir::symbol_table.at(for_stride).first;
                    } else {
                        stride = builder.create<mlir::arith::ConstantIndexOp>(location, std::stoi(for_stride->unparseToString()));
                        upir::symbol_table[for_stride] = std::make_pair(stride, for_stride_symbol);
                    }
                }

                mlir::ValueRange loop_value = {};
                mlir::scf::ForOp loop = builder.create<mlir::scf::ForOp>(location, lower_bound, upper_bound, stride, loop_value);
                mlir::Region &loop_body = loop.getLoopBody();
                mlir::Block &loop_block = loop_body.front();
                builder.setInsertionPointToStart(&loop_block);

                if (isSgBasicBlock(for_body)) {
                    convert_basic_block(builder, isSgBasicBlock(for_body));
                } else {
                    convert_statement(builder, for_body);
                }
                builder.setInsertionPointAfter(loop);
                break;
            }
        default:
            {
                ;
            }
    }
}

void convert_basic_block(mlir::OpBuilder& builder, SgBasicBlock* block) {

    SgStatementPtrList& statements = block->get_statements();
    SgStatementPtrList::iterator iter;
    for (iter = statements.begin(); iter != statements.end(); iter++) {
        SgStatement* node = isSgStatement(*iter);
        if (node != NULL) {
            convert_statement(builder, node);
        }
    }
}


class UpirSageAST : public AstTopDownProcessing<InheritedAttribute> {


    public:
        UpirSageAST(mlir::ModuleOp& __root, mlir::OpBuilder& __builder) : root(__root), builder(__builder) { };
        mlir::ModuleOp& get_root() { return root; };
        void set_root(mlir::ModuleOp& __root) { root = __root; };
        mlir::OpBuilder& get_builder() { return builder; };
        void set_builder(mlir::OpBuilder& __builder) { builder = __builder; };

    protected:
        mlir::ModuleOp root;
        mlir::OpBuilder builder;
        virtual InheritedAttribute evaluateInheritedAttribute(SgNode*, InheritedAttribute) override;
};

InheritedAttribute UpirSageAST::evaluateInheritedAttribute(SgNode* node, InheritedAttribute attribute) {
    if (isSgStatement(node)) {
        std::cout << "SgNode: " << node->sage_class_name() << " at line: " << node->get_startOfConstruct()->get_line() << "\n";
    }
    switch (node->variantT()) {
        case V_SgFunctionDefinition:
            {
                if (node->get_file_info()->get_line() == 0) {
                    return InheritedAttribute(NULL, false, attribute.depth+1, node);
                }
                SgFunctionDefinition* target = isSgFunctionDefinition(node);
                llvm::SmallVector<mlir::Type, 4> arg_types;

                SgInitializedNamePtrList function_parameters = target->get_declaration()->get_args();
                SgInitializedNamePtrList::const_iterator iter;
                for (iter = function_parameters.begin(); iter != function_parameters.end(); iter++) {
                    mlir::Type type = nullptr;
                    SgInitializedName* symbol = *iter;
                    SgType* symbol_type = symbol->get_type();
                    if (isSgPointerType(symbol_type)) {
                        type = mlir::UnrankedMemRefType::get(builder.getI32Type(), 8);
                    } else {
                        type = builder.getI32Type();
                    }
                    arg_types.push_back(type);
                    assert(upir::symbol_table.count(symbol) == 0);
                    upir::symbol_table[symbol] = std::make_pair(nullptr, symbol);
                }

                mlir::Location location = builder.getUnknownLoc();

                std::cout << "Prepare base function parameters...." << std::endl;
                auto func_type = builder.getFunctionType(arg_types, llvm::None);

                std::cout << "Prepare base function name...." << std::endl;
                llvm::StringRef func_name = std::string(target->get_declaration()->get_name().getString());

                std::cout << "Create a base function...." << std::endl;
                mlir::FuncOp func = mlir::FuncOp::create(location, func_name, func_type);

                std::cout << "Create the body of base function...." << std::endl;
                mlir::Block &entryBlock = *func.addEntryBlock();

                // update the actual mlir::Value of each function parameter
                int i = 0;
                for (iter = function_parameters.begin(); iter != function_parameters.end(); iter++) {
                    SgInitializedName* symbol = *iter;
                    upir::symbol_table[symbol] = std::make_pair(entryBlock.getArgument(i), symbol);
                    i++;
                }
                builder.setInsertionPointToStart(&entryBlock);

                SgBasicBlock* func_body = target->get_body();
                convert_basic_block(builder, func_body);
                builder.setInsertionPointAfter(func);

                root.push_back(func);

                return InheritedAttribute(NULL, true, attribute.depth+1, node);
            }
        default:
            {
                if (isSgStatement(node)) {
                    //std::cout << "======  Meet a new SgNode: " << node->sage_class_name() << " ======\n";
                };
                return InheritedAttribute(attribute.frontier, true, attribute.depth+1, node);
            }
    };

};

mlir::ModuleOp generate_mlir(mlir::MLIRContext& context, SgProject* project) {

    ROSE_ASSERT(project != NULL);
    upir::g_project = project;

    mlir::OpBuilder builder = mlir::OpBuilder(&context);
    mlir::ModuleOp project_module = mlir::ModuleOp::create(builder.getUnknownLoc());

    UpirSageAST upir_ast = UpirSageAST(project_module, builder);
    InheritedAttribute attribute(NULL, true, 0, NULL);
    upir_ast.traverseInputFiles(project, attribute);

    return project_module;
};

