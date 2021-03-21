#include <vector>
#include <string>
#include "rose.h"

class InheritedAttribute {

    public:
        SgOmpFlowGraphNode* frontier = NULL;
        SgNode* expr_parent = NULL;
        bool to_explore = false;
        size_t depth = 0;

        InheritedAttribute(SgOmpFlowGraphNode* __frontier, bool __to_explore, size_t __depth, SgNode* __expr_parent) : frontier(__frontier), to_explore(__to_explore), depth(__depth), expr_parent(__expr_parent) { };

};

class OmpFlowGraph : public AstTopDownProcessing<InheritedAttribute> {

    protected:
        SgOmpFlowGraphNode* root = NULL;
        SgNode* cursor = NULL;
        std::vector<SgOmpFlowGraphNode* > task_nodes;
        bool has_serial_node_candidate = false;
        size_t previous_depth = 0;
        virtual InheritedAttribute evaluateInheritedAttribute(SgNode*, InheritedAttribute);

    public:
        SgOmpFlowGraphNode* get_root() { return root; };
        std::vector<SgOmpFlowGraphNode* >* get_nodes() { return &task_nodes; };
        void add_task_node(SgOmpFlowGraphNode* __node) { task_nodes.push_back(__node); };
        void set_previous_depth(size_t __depth) { previous_depth = __depth; };
        size_t get_previous_depth() { return previous_depth; };

};

SgOmpFlowGraphNode* generate_graph(SgProject*);

