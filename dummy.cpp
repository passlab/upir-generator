#include <iostream>
#include "rose.h"
#include "flow_graph.h"
extern void dummy_visualize(Node*);

using namespace SageBuilder;
using namespace SageInterface;

int main (int argc, char *argv[]) {

    // generate the REX AST
    SgProject* project = frontend(argc, argv);
    SgGlobal* global = getFirstGlobalScope(project);

    SgFunctionDeclaration* main_func = findMain(project);
    SgBasicBlock* body = main_func->get_definition()->get_body();
  
    // generate a dummy task graph
    Node* root = generate_dummy_graph();

    // visualize the graph to a DOT file
    dummy_visualize(root);

    return 0;
}

