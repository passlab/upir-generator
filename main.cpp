#include <iostream>
#include "rose.h"
extern SgOmpFlowGraphNode* generate_graph(SgProject*);
extern void visualize(SgOmpFlowGraphNode*);

using namespace SageBuilder;
using namespace SageInterface;

int main (int argc, char *argv[]) {

    // generate the REX AST
    SgProject* project = frontend(argc, argv);

    // generate the task graph
    SgOmpFlowGraphNode* root = generate_graph(project);

    std::list<SgNode* > children = root->get_children();

    printf("Check the task graph....\n");
    while (children.size()) {
        SgNode* node = ((SgOmpFlowGraphNode*)children.front())->get_node();
        std::cout << "SgNode: " << node->sage_class_name() << " at line: " << node->get_startOfConstruct()->get_line() << "\n";
        children = ((SgOmpFlowGraphNode*)children.front())->get_children();
    };

    // visualize the graph to a DOT file
    //visualize(root);

    return 0;
}

