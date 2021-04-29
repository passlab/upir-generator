#include <iostream>
#include "flow_graph.h"

InheritedAttribute OmpFlowGraph::evaluateInheritedAttribute(SgNode* node, InheritedAttribute attribute) {

    if (isSgStatement(node)) {
        std::cout << "SgNode: " << node->sage_class_name() << " at line: " << node->get_startOfConstruct()->get_line() << "\n";
    }
    else {
    //    std::cout << "SgNode: " << node->sage_class_name() << " has no line number.\n";
    };
    switch (node->variantT()) {
        case V_SgFunctionDefinition:
            {
                std::cout << "Add a function definition.\n";
                SgOmpFlowGraphSerialNode* graph_node = new SgOmpFlowGraphSerialNode("", node);
                root = graph_node;

                return InheritedAttribute(graph_node, true, attribute.depth+1, node);
            }
        case V_SgAddOp:
            {
                std::cout << "Found a SgAddOp node.\n";

                return InheritedAttribute(NULL, false, attribute.depth+1, node);
            }
        case V_SgVariableDeclaration:
        case V_SgExprStatement:
            {
                // reset the serial flag if the last explored expr statement and the current one have different parents
                if (cursor != attribute.expr_parent) {
                    has_serial_node_candidate = false;
                };
                // omit the current node and its sub-tree if it doesn't need to be explored
                SgNode* parent = node->get_parent();
                if (attribute.frontier == NULL || has_serial_node_candidate || (isSgForStatement(parent)) && (isSgForStatement(parent)->get_loop_body() != node)) {
                    return InheritedAttribute(NULL, false, attribute.depth+1, node);
                };

                SgOmpFlowGraphSerialNode* graph_node = new SgOmpFlowGraphSerialNode("", node);
                has_serial_node_candidate = true;

                std::list<SgNode* > children = attribute.frontier->get_children();
                children.push_back(graph_node);
                attribute.frontier->set_children(children);

                std::list<SgNode* > parents;
                parents.push_back(attribute.frontier);
                graph_node->set_parents(parents);
                std::cout << "Add a variable declaration or expression statement.\n";

                set_previous_depth(attribute.depth+1);
                cursor = attribute.expr_parent;
                return InheritedAttribute(NULL, false, attribute.depth+1, node);
            }
        case V_SgOmpForStatement:
        case V_SgOmpParallelStatement:
        case V_SgOmpBarrierStatement:
            {
                std::cout << "Add an omp pragma.\n";

                SgOmpFlowGraphTaskNode* graph_node = new SgOmpFlowGraphTaskNode("", node);
                has_serial_node_candidate = false;

                // set up an average cost
                graph_node->set_cost(4000);

                std::list<SgNode* > children = attribute.frontier->get_children();
                children.push_back(graph_node);
                attribute.frontier->set_children(children);

                std::list<SgNode* > parents;
                parents.push_back(attribute.frontier);
                graph_node->set_parents(parents);

                return InheritedAttribute(graph_node, true, attribute.depth+1, node);
            }
        case V_SgForStatement:
            {
                std::cout << "Add a for loop statement.\n";

                SgOmpFlowGraphSerialNode* graph_node = new SgOmpFlowGraphSerialNode("", node);
                has_serial_node_candidate = false;

                SgInitializedName * orig_index = NULL;
                SgExpression* orig_lower = NULL;
                SgExpression* orig_upper = NULL;
                SgExpression* orig_stride = NULL;
                bool isIncremental = true;
                bool is_canonical = false;
                is_canonical = SageInterface::isCanonicalForLoop(isSgForStatement(node), &orig_index, &orig_lower, &orig_upper, &orig_stride, NULL, &isIncremental);
                std::string lower_bound = orig_lower->unparseToString();
                std::string upper_bound = orig_upper->unparseToString();
                std::string stride = orig_stride->unparseToString();
                int iteration_amount = (std::stoi(upper_bound) - std::stoi(lower_bound))/std::stoi(stride);
                std::cout << "Lower bound: " << lower_bound << std::endl;
                std::cout << "Upper bound: " << upper_bound << std::endl;
                std::cout << "Stride: " << stride << std::endl;
                std::cout << "Iteration amount: " << iteration_amount << std::endl;

                // set up an average cost
                graph_node->set_cost(iteration_amount);

                std::list<SgNode* > children = attribute.frontier->get_children();
                children.push_back(graph_node);
                attribute.frontier->set_children(children);

                std::list<SgNode* > parents;
                parents.push_back(attribute.frontier);
                graph_node->set_parents(parents);

                return InheritedAttribute(graph_node, true, attribute.depth+1, node);
            }
        case V_SgBasicBlock:
            {
                std::cout << "Skip the basic block.\n";
                has_serial_node_candidate = false;

                return InheritedAttribute(attribute.frontier, true, attribute.depth+1, node);
            }
        case V_SgForInitStatement:
            {
                return InheritedAttribute(NULL, false, attribute.depth+1, node);
            }
        default:
            {
                if (isSgStatement(node)) {
                    std::cout << "======  Meet a new SgNode: " << node->sage_class_name() << " ======\n";
                };
                return InheritedAttribute(attribute.frontier, true, attribute.depth+1, node);
            }
    };

};

SgOmpFlowGraphNode* generate_graph(SgProject* project) {

    ROSE_ASSERT(project != NULL);

    SgOmpFlowGraphSerialNode* graph_node = new SgOmpFlowGraphSerialNode("", project);
    InheritedAttribute attribute(graph_node, true, 0, NULL);
    OmpFlowGraph task_graph;
    task_graph.traverseInputFiles(project, attribute);

    return task_graph.get_root();
};


