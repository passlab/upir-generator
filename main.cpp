#include <iostream>
#include "rose.h"
extern SgOmpFlowGraphNode* generate_graph(SgProject*);
extern void visualize_node(SgOmpFlowGraphNode*);
extern void visualize(SgOmpFlowGraphNode*);

using namespace SageBuilder;
using namespace SageInterface;

void visit(SgOmpFlowGraphNode* root) {

    std::cout << "SgNode: " << root->get_node()->sage_class_name() << " at line: " << root->get_node()->get_startOfConstruct()->get_line() << "\n";

    // TODO: visualize a single graph node
    visualize_node(root);

    std::list<SgNode* > children = root->get_children();
    if (children.size()) {
        std::list<SgNode* >::iterator iter;
        for (iter = children.begin(); iter != children.end(); iter++) {
            SgOmpFlowGraphNode* child = ((SgOmpFlowGraphNode*)(*iter));
            SgNode* node = child->get_node();
            visit(child);
        };
    };
}


int main (int argc, char *argv[]) {

    // generate the REX AST
    SgProject* project = frontend(argc, argv);

    // generate the task graph
    SgOmpFlowGraphNode* root = generate_graph(project);

    // list all statements in the task graph
    printf("Check the parallel flow graph....\n");
    visit(root);

    // TODO: visualize the whole graph and store it to a DOT file
    visualize(root);

    return 0;
}

