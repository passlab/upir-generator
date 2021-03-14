#include <iostream>
#include "flow_graph.h"

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

