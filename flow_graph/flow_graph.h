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

// TODO: add REX SgNode enums to indicates type and label
enum NodeType {

    GenericNodeType = 0,
    SerialNodeType = 1,
    TaskNodeType = 2,

};

// TODO: use REX SgNode and APIs to implement Node class,
// such as SgOmpFlowGraphBaseNode, SgOmpFlowGraphSerialNode, and SgOmpFlowGraphTaskNode
class Node {

    protected:
        std::vector<Node* > parents;
        std::vector<Node* > children;
        // TODO: use REX SgNode enums instead
        NodeType type = GenericNodeType;
        std::string label = "";

    public:
        Node(std::string __label = "") { label == __label; };
        std::vector<Node* >* get_parents() { return &parents; };
        void add_parent(Node* __parent) { parents.push_back(__parent); };
        std::vector<Node* >* get_children() { return &children; };
        void add_child(Node* __child) { children.push_back(__child); };
        bool has_parent() { return parents.size() > 0; };
        bool has_children() { return children.size() > 0; };
        NodeType get_type() { return type; };
        void set_type(NodeType __type) { type = __type; };
        std::string get_label() { return label; };
        void set_label(std::string __label) { label = __label; };

};

class SerialNode : public Node {

    protected:
        std::vector<std::string> data;

    public:
        SerialNode(std::string __label = "") : Node(__label) { type = SerialNodeType; };
        std::vector<std::string>* get_data() { return &data; };
        void add_data(std::string __data) { data.push_back(__data); };

};

class TaskNode : public SerialNode {

    protected:
        int num_threads = 0;
        Node* head = NULL;
        Node* tail = NULL;

    public:
        TaskNode(std::string __label = "", int __num_threads = 0, Node* __head = NULL, Node* __tail = NULL) : SerialNode(__label) { num_threads = __num_threads; head = __head; tail = __tail; type = TaskNodeType; };
        int get_num_threads() { return num_threads; };
        void set_num_threads(int __num_threads) { num_threads = __num_threads; };
        Node* get_head() { return head; };
        void set_head(Node* __head) { head = __head; };
        Node* get_tail() { return tail; };
        void set_tail(Node* __tail) { tail = __tail; };

};

SgOmpFlowGraphNode* generate_graph(SgProject*);
Node* generate_dummy_graph();
