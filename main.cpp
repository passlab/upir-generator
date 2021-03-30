#include <iostream>
#include "rose.h"
extern SgOmpFlowGraphNode* generate_graph(SgProject*);
extern void visualize_node(SgOmpFlowGraphNode*);
extern void visualize(SgOmpFlowGraphNode*);

using namespace SageBuilder;
using namespace SageInterface;


int main (int argc, char *argv[]) {

    // generate the REX AST
    SgProject* project = frontend(argc, argv);

    // generate the task graph
    SgOmpFlowGraphNode* root = generate_graph(project);

    // list all statements in the task graph
    printf("Check the parallel flow graph....\n");
    //visit(root);

    // TODO: visualize the whole graph and store it to a DOT file
    visualize(root);

    return 0;
}

