#include <iostream>
#include <algorithm>
#include "visualization.h"


std::string get_node_header(std::string body, std::string type) {
    std::string node_label;
    std::string node_shape;
    std::string node_header;
    switch(std::stoi(type)) {
      case 164:
        node_label = "base";
	node_shape = "box";
        break;
      case 700:
        node_label = "base";
	node_shape = "box";
        break;
      case 142:
        node_label = "serial";
	node_shape = "box";
        break;
      case 308:
        node_label = "omp_parallel";
	node_shape = "doublecircle";
        break;
      case 294:
        node_label = "omp_barrier";
	node_shape = "doublecircle";
        break;
      case 302:
        node_label = "omp_for";
	node_shape = "ellipse";
        break;
      case 154:
        node_label = "for";
	node_shape = "ellipse";
        break;
    };

    node_header = "    " + \
	          body +\
                  "[label = " + node_label + "_node" + " " +  "shape = " + node_shape + "]" + \
                  "\n";
    return node_header;
};

std::string get_list_element(std::list<std::string> list, int i) {
    if (list.size()) { 
        std::list<std::string>::iterator k = list.begin();
        for (int c = 0; c < i; c++) {
            ++k;
        };
	return *k;
    }
    // TODO: fix the missing return statement
    return "";
};

std::string get_node_types(SgOmpFlowGraphNode* node) {
    std::string node_types;
    switch(node->get_node()->variantT()) {
      case V_SgFunctionDefinition:
        node_types = "base";
        break;
      case V_SgVariableDeclaration:
        node_types = "base";
        break;
      case V_SgExprStatement:
        node_types = "serial";
        break;
      case V_SgOmpParallelStatement:
        node_types = "omp_parallel";
        break;
      case V_SgOmpBarrierStatement:
        node_types = "omp_barrier";
        break;
      case V_SgOmpForStatement:
        node_types = "omp_for";
        break;
      case V_SgForStatement:
        node_types = "for";
        break;
    };
    return node_types;
};

// TODO: visualize a single node
void visualize_node(SgOmpFlowGraphNode* node) {

    // implementation

};

void visit(SgOmpFlowGraphNode* node, std::list<std::string> *body, std::list<std::string> *layers, std::list<std::string> *types) {
    //std::cout << "SgNode: " << node->get_node()->sage_class_name() << " at line: " << node->get_node()->get_startOfConstruct()->get_line() << "\n";
    int layer;
    std::list<SgNode* > children = node->get_children();
    std::list<SgNode* > parents = node->get_parents();
    if (!parents.size()) {
        body->push_back(get_node_types(node)+std::to_string(node->get_node()->get_startOfConstruct()->get_line()));
    };

    types->push_back(std::to_string(node->get_node()->variantT()));

    std::list<SgNode* > tmp_parents = node->get_parents();
    int tmp_layer = 1;
    while(tmp_parents.size()) {
        SgOmpFlowGraphNode* tmp_parent = (SgOmpFlowGraphNode*)tmp_parents.front();
	if (tmp_parent->get_node()->variantT() == V_SgOmpParallelStatement) tmp_layer++;
	tmp_parents = tmp_parent->get_parents();
    };
    layers->push_back(std::to_string(tmp_layer));

    if (children.size()) {
        std::list<SgNode* >::iterator iter;
        for (iter = children.begin(); iter != children.end(); iter++) {
            SgOmpFlowGraphNode* child = ((SgOmpFlowGraphNode*)(*iter));
            body->push_back(get_node_types(child)+std::to_string(child->get_node()->get_startOfConstruct()->get_line()));

	    visit(child, body, layers, types);
        };
    };
};

void draw(std::list<std::string> body, std::list<std::string> layers, std::list<std::string> types) {
/*
    if (body.size()) {
        std::list<std::string>::iterator i;
        for (i = body.begin(); i != body.end(); i++) {
            std::cout << *i << " ";
        };
    };
    std::cout << "\n";
    if (layers.size()) {
        std::list<std::string>::iterator j;
	for (j = layers.begin(); j != layers.end(); j++) {
            std::cout << *j << "\t";
        };
    };
    std::cout << "\n";
    if (types.size()) {
        std::list<std::string>::iterator k;
        for (k = types.begin(); k != types.end(); k++) {
            std::cout << *k << "\t";
        };
    };
    std::cout << "\n"; 
 */
// draw the edges    
    if (body.size()) {
	std::string headers; 
	std::string lines[4];
        int tmp;
	int entries[2];
	for (int e = 0; e < 2; e++) entries[e] = 0;
	lines[0].append("    " + get_list_element(body, 0));
	std::cout << get_node_header(get_list_element(body, 0), get_list_element(types, 0));
	for (int q = 1; q < body.size(); q++) {
            tmp = std::stoi(get_list_element(layers, q));
	    //std::cout << tmp << "\n";
	    if (tmp > 2) {
		if (std::stoi(get_list_element(layers, q-1)) == tmp-1) {
		    lines[3].append("    " + get_list_element(body, q-1));
		    lines[2].append("    " + get_list_element(body, q-1));
		};
		if (std::stoi(get_list_element(types, q)) == 294) {
                    lines[3].append("->" + get_list_element(body, q));
                    lines[2].append("->" + get_list_element(body, q));
		} else {
	            lines[3].append("->p2_" + get_list_element(body, q));
		    std::cout << get_node_header("p2_"+get_list_element(body, q), get_list_element(types, q));
	            lines[2].append("->p3_" + get_list_element(body, q));
		    std::cout << get_node_header("p3_"+get_list_element(body, q), get_list_element(types, q));
	        };
		if (std::stoi(get_list_element(types, q)) == 154 && tmp == 3) {
		    lines[3].append("->p3_omp_for_barrier");
		    std::cout << get_node_header("p3_omp_for_barrier", "302");
		    lines[2].append("->p2_omp_for_barrier");
		    std::cout << get_node_header("p2_omp_for_barrier", "302");
		}
		if (std::stoi(get_list_element(layers, q+1)) == tmp-1) {
		    lines[3].append("->" + get_list_element(body, q+1));
		    lines[2].append("->" + get_list_element(body, q+1));
		};
            };
	    if (tmp > 1) {
		if (std::stoi(get_list_element(layers, q-1)) == tmp-1) lines[1].append("    " + get_list_element(body, q-1));
		if (std::stoi(get_list_element(types, q)) == 294) {
	            lines[1].append("->" + get_list_element(body, q));
                } else {
        	    lines[1].append("->p1_" + get_list_element(body, q));
		    std::cout << get_node_header("p1_"+get_list_element(body, q), get_list_element(types, q));
		};
		if (std::stoi(get_list_element(types, q)) == 154 && tmp == 2) {
		    lines[1].append("->p1_omp_for_barrier");
		    std::cout << get_node_header("p1_omp_for_barrier", "302");
		}
		if (std::stoi(get_list_element(layers, q+1)) == tmp-1) lines[1].append("->" + get_list_element(body, q+1));
	    };
	    if (tmp > 0) {
	        lines[0].append("->" + get_list_element(body, q));
		std::cout << get_node_header(get_list_element(body, q), get_list_element(types, q));
		if (std::stoi(get_list_element(types, q)) == 154) {
		    lines[0].append("->omp_for_barrier");
	            std::cout << get_node_header("omp_for_barrier", "302");
		};
	    };
	};
        
        for (int l = 0; l < 4; l++) std::cout << lines[l] << "\n";
    };
};

// TODO: visualize the whole graph starting from a given node
void visualize(SgOmpFlowGraphNode* node) {
    std::list<std::string> body;
    std::list<std::string> layers;
    std::list<std::string> types;
// write to DOT file
    std::cout << "[DOT file starts]\n";
    std::cout << "digraph G {" << "\n";
    
    visit(node, &body, &layers, &types); 
    draw(body, layers, types);

    std::cout << "}" << "\n"; 
    std::cout << "[DOT file ends]\n";
};

