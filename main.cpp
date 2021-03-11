#include <iostream>
#include "rose.h"

using namespace SageBuilder;
using namespace SageInterface;

int main (int argc, char *argv[]) {
  SgProject* project = frontend(argc, argv);
  SgGlobal* global = getFirstGlobalScope(project);

  SgFunctionDeclaration* main_func= findMain(project);
  SgBasicBlock* body= main_func->get_definition()->get_body();
  
  AstTests::runAllTests(project);
  project->unparse();

  return 0;
}

