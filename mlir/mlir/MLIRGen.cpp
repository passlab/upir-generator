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

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "rose.h"

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

void convert_statement(mlir::OpBuilder& builder, SgStatement* node) {

    mlir::Location location = builder.getUnknownLoc();
    switch (node->variantT()) {
        case V_SgOmpParallelStatement:
            {
                SgOmpParallelStatement* target = isSgOmpParallelStatement(node);
                std::cout << "Insert a SPMD region...." << std::endl;
                mlir::Value num_threads = nullptr;
                if (OmpSupport::hasClause(target, V_SgOmpNumThreadsClause)) {
                    Rose_STL_Container<SgOmpClause*> num_threads_clauses = OmpSupport::getClause(target, V_SgOmpNumThreadsClause);
                    SgOmpNumThreadsClause* num_threads_clause = isSgOmpNumThreadsClause(num_threads_clauses[0]);
                    std::string num_threads_string = num_threads_clause->get_expression()->unparseToString();
                    int32_t num_thread_value = std::stoi(num_threads_string);
                    num_threads = builder.create<mlir::ConstantIntOp>(location, num_thread_value, 32);
                }
                mlir::pirg::SpmdOp spmd = builder.create<mlir::pirg::SpmdOp>(location, num_threads, nullptr, mlir::ValueRange(), mlir::ValueRange(), nullptr);
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
        case V_SgOmpTargetStatement:
            {
                SgOmpTargetStatement* target = isSgOmpTargetStatement(node);
                std::cout << "Insert a target region...." << std::endl;
                mlir::StringAttr device = builder.getStringAttr(llvm::StringRef("nvptx"));
                mlir::pirg::TaskOp task_target = builder.create<mlir::pirg::TaskOp>(location, nullptr, device, nullptr, mlir::ValueRange(), nullptr);
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
                mlir::pirg::TaskOp task = builder.create<mlir::pirg::TaskOp>(location, nullptr, nullptr, nullptr, mlir::ValueRange(), nullptr);
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
            {
                SgForStatement* target = isSgForStatement(node);
                std::cout << "Insert a for loop...." << std::endl;

                SgInitializedName* for_index = NULL;
                SgExpression* for_lower = NULL;
                SgExpression* for_upper = NULL;
                SgExpression* for_stride = NULL;
                bool isIncremental = true;
                bool is_canonical = false;
                is_canonical = SageInterface::isCanonicalForLoop (target, &for_index, &for_lower, &for_upper, &for_stride, NULL, &isIncremental);
                ROSE_ASSERT(is_canonical == true);
                int32_t lower_value = std::stoi(for_lower->unparseToString());
                int32_t upper_value = std::stoi(for_upper->unparseToString());
                int32_t stride_value = std::stoi(for_stride->unparseToString());

                mlir::Value lower_bound = builder.create<mlir::ConstantIndexOp>(location, lower_value);
                mlir::Value upper_bound = builder.create<mlir::ConstantIndexOp>(location, upper_value);
                mlir::Value step = builder.create<mlir::ConstantIndexOp>(location, stride_value);
                mlir::ValueRange loop_value = {};
                mlir::scf::ForOp loop = builder.create<mlir::scf::ForOp>(location, upper_bound, lower_bound, step, loop_value);
                mlir::Region &loop_body = loop.getLoopBody();
                mlir::Block &loop_block = loop_body.front();
                builder.setInsertionPointToStart(&loop_block);

                SgStatement* for_body = target->get_loop_body();
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


class PirgSageAST : public AstTopDownProcessing<InheritedAttribute> {


    public:
        PirgSageAST(mlir::ModuleOp& __root, mlir::OpBuilder& __builder) : root(__root), builder(__builder) { };
        mlir::ModuleOp& get_root() { return root; };
        void set_root(mlir::ModuleOp& __root) { root = __root; };
        mlir::OpBuilder& get_builder() { return builder; };
        void set_builder(mlir::OpBuilder& __builder) { builder = __builder; };

    protected:
        mlir::ModuleOp root;
        mlir::OpBuilder builder;
        virtual InheritedAttribute evaluateInheritedAttribute(SgNode*, InheritedAttribute) override;
};

InheritedAttribute PirgSageAST::evaluateInheritedAttribute(SgNode* node, InheritedAttribute attribute) {

    if (isSgStatement(node)) {
        std::cout << "SgNode: " << node->sage_class_name() << " at line: " << node->get_startOfConstruct()->get_line() << "\n";
    }
    switch (node->variantT()) {
        case V_SgFunctionDefinition:
            {
                std::cout << "Add a function definition.\n";
                SgFunctionDefinition* target = isSgFunctionDefinition(node);

                mlir::Location location = builder.getUnknownLoc();

                std::cout << "Prepare base function parameters...." << std::endl;
                llvm::ArrayRef<std::unique_ptr<llvm::StringRef>> args;
                llvm::SmallVector<mlir::Type, 4> arg_types(args.size(), builder.getNoneType());
                auto func_type = builder.getFunctionType(arg_types, llvm::None);
                llvm::ArrayRef<std::pair<mlir::Identifier, mlir::Attribute> > attrs = {};

                std::cout << "Prepare base function name...." << std::endl;
                llvm::StringRef func_name = std::string(target->get_declaration()->get_name().getString());

                std::cout << "Create a base function...." << std::endl;
                mlir::FuncOp func = mlir::FuncOp::create(location, func_name, func_type, attrs);

                std::cout << "Create the body of base function...." << std::endl;
                mlir::Block &entryBlock = *func.addEntryBlock();

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

    mlir::OpBuilder builder = mlir::OpBuilder(&context);
    mlir::ModuleOp project_module = mlir::ModuleOp::create(builder.getUnknownLoc());

    PirgSageAST pirg_ast = PirgSageAST(project_module, builder);
    InheritedAttribute attribute(NULL, true, 0, NULL);
    pirg_ast.traverseInputFiles(project, attribute);

    return project_module;
};

