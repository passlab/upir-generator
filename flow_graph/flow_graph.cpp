#include <iostream>
#include "flow_graph.h"

OmpFlowGraph::OmpFlowGraph() {

};

void OmpFlowGraph::visit(SgNode* node) {

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
                root = new SgOmpFlowGraphSerialNode("", node);
                cursor = root;
                add_task_node(root);
                break;
            }
        case V_SgVariableDeclaration:
        case V_SgExprStatement:
            {
                SgNode* parent = node->get_parent();
                if (!isSgForStatement(node) && !has_serial_node_candidate) {
                    SgOmpFlowGraphSerialNode* graph_node = new SgOmpFlowGraphSerialNode("", node);
                    add_task_node(graph_node);
                    std::list<SgNode* > parents;
                    parents.push_back(cursor);
                    std::list<SgNode* > children;
                    children.push_back(graph_node);
                    graph_node->set_parents(parents);
                    cursor->set_children(children);
                    cursor = graph_node;
                    std::cout << "Add a variable declaration or expression statement.\n";
                    has_serial_node_candidate = true;
                };
                break;
            }
        case V_SgPragmaDeclaration:
            {
                std::cout << "Add a #pragma.\n";
                SgOmpFlowGraphTaskNode* graph_node = new SgOmpFlowGraphTaskNode("", node);
                has_serial_node_candidate = false;
                add_task_node(graph_node);
                std::list<SgNode* > parents;
                parents.push_back(cursor);
                std::list<SgNode* > children;
                children.push_back(graph_node);
                graph_node->set_parents(parents);
                cursor->set_children(children);
                cursor = graph_node;
                break;
            }
        case V_SgForStatement:
            {
                std::cout << "Add a for loop statement.\n";
                SgOmpFlowGraphSerialNode* graph_node = new SgOmpFlowGraphSerialNode("", node);
                has_serial_node_candidate = false;
                add_task_node(graph_node);
                std::list<SgNode* > parents;
                parents.push_back(cursor);
                std::list<SgNode* > children;
                children.push_back(graph_node);
                graph_node->set_parents(parents);
                cursor->set_children(children);
                cursor = graph_node;
                break;
            }
        default:
            {

            }
    };

};

SgOmpFlowGraphNode* generate_graph(SgProject* project) {

    ROSE_ASSERT(project != NULL);

    OmpFlowGraph task_graph;
    task_graph.traverseInputFiles(project, preorder);

    return task_graph.get_root();
};

Node* generate_dummy_graph () {

    // start of program
    SerialNode* start = new SerialNode("main");

    SerialNode* s1 = new SerialNode("S1");
    s1->add_parent(start);
    start->add_child(s1);

    // start of parallel region, specify 6 threads
    TaskNode* omp_parallel = new TaskNode("omp parallel", 6);
    omp_parallel->add_parent(s1);
    s1->add_child(omp_parallel);
    omp_parallel->add_data("a[5]");
    omp_parallel->add_data("b[10]");
    omp_parallel->add_data("c");
    omp_parallel->add_data("d");

    SerialNode* s2 = new SerialNode("S2");
    s2->add_parent(omp_parallel);
    omp_parallel->add_child(s2);
    omp_parallel->set_head(s2);

    TaskNode* omp_for = new TaskNode("omp for", 1);
    omp_for->add_parent(s2);
    s2->add_child(omp_for);
    omp_for->add_data("a[5]");
    omp_for->add_data("b[10]");

    // for loop body and implicit barrier
    SerialNode* s3 = new SerialNode("S3");
    s3->add_parent(omp_for);
    omp_for->add_child(s3);
    omp_for->set_head(s3);
    omp_for->set_tail(s3);

    SerialNode* s4 = new SerialNode("S4");
    s4->add_parent(s3);
    s3->add_child(s4);

    // explicit barrier
    TaskNode* omp_barrier = new TaskNode("omp barrier");
    omp_barrier->add_parent(s4);
    s4->add_child(omp_barrier);

    SerialNode* s5 = new SerialNode("S5");
    s5->add_parent(omp_barrier);
    omp_barrier->add_child(s5);
    omp_parallel->set_tail(s5);
    // the end of parallel region

    // the end of program
    SerialNode* end = new SerialNode("EOF");
    end->add_parent(s5);
    s5->add_child(end);

    return start;

};

