#include "rose.h"

class ParallelData {
    private:
        SgVariableSymbol* symbol = NULL;
        std::string sharing_property = "";
        std::string sharing_visibility = "";
        std::string mapping_property = "";
        std::string mapping_visibility = "";
        std::string data_access = "";

    public:
        ParallelData(SgVariableSymbol* __symbol, std::string __sharing_property = "", std::string __sharing_visibility = "", std::string __mapping_property = "", std::string __mapping_visibility = "", std::string __data_access = "") : symbol(__symbol), sharing_property(__sharing_property), sharing_visibility(__sharing_visibility), mapping_property(__mapping_property), mapping_visibility(__mapping_visibility), data_access(__data_access) {};
        SgVariableSymbol* get_symbol() { return symbol; };
        void set_sharing_property(std::string __sharing_property) { sharing_property = __sharing_property; };
        std::string get_sharing_property() { return sharing_property; };
        void set_sharing_visibility(std::string __sharing_visibility) { sharing_visibility = __sharing_visibility; };
        std::string get_sharing_visibility() { return sharing_visibility; };
        void set_mapping_property(std::string __mapping_property) { mapping_property = __mapping_property; };
        std::string get_mapping_property() { return mapping_property; };
        void set_mapping_visibility(std::string __mapping_visibility) { mapping_visibility = __mapping_visibility; };
        std::string get_mapping_visibility() { return mapping_visibility; };
        void set_data_access(std::string __data_access) { data_access = __data_access; };
        std::string get_data_access() { return data_access; };

        void output() {
            std::cout << "Variable symbol: " << symbol->get_name() << "\n";
            std::cout << "\tSharing property: " << sharing_property << "\n";
            std::cout << "\tSharing visibility: " << sharing_visibility << "\n";
            std::cout << "\tMapping property: " << mapping_property << "\n";
            std::cout << "\tMapping visibility: " << mapping_visibility << "\n";
            std::cout << "\tData access: " << data_access << "\n";
        };
};

std::map<SgVariableSymbol *, ParallelData *> analyze_parallel_data(SgSourceFile*);
