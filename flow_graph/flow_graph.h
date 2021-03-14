#include <vector>
#include <string>

enum NodeType {

    GenericNodeType = 0,
    SerialNodeType = 1,
    TaskNodeType = 2,

};

class Node {

    protected:
        std::vector<Node* > parents;
        std::vector<Node* > children;
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

Node* generate_graph();
Node* generate_dummy_graph();
