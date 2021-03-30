#include "flow_graph.h"

std::string get_node_header(std::string body, std::string type);
std::string get_list_element(std::list<std::string> list, int i);
std::string get_node_types(SgOmpFlowGraphNode* node);

void visit(SgOmpFlowGraphNode* node, std::list<std::string> *body, std::list<std::string> *layers, std::list<std::string> *types);
void draw(std::list<std::string> body, std::list<std::string> layers, std::list<std::string> *types);

void visualize_node(SgOmpFlowGraphNode* node);
void visualize(SgOmpFlowGraphNode* node);
