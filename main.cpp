#include <iostream>
#include "rose.h"
#include "flow_graph.h"

using namespace SageBuilder;
using namespace SageInterface;

int main (int argc, char *argv[]) {

    // generate the REX AST
    SgProject* project = frontend(argc, argv);
    SgGlobal* global = getFirstGlobalScope(project);

    SgFunctionDeclaration* main_func= findMain(project);
    SgBasicBlock* body= main_func->get_definition()->get_body();
  
    AstTests::runAllTests(project);
    project->unparse();

    // generate a dummy task graph
    Node* root = generate_dummy_graph();

    return 0;
}

