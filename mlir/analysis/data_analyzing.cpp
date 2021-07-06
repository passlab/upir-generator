#include "data_analyzing.h"

std::map<SgVariableSymbol *, ParallelData *> parallel_data;

// Patch up private variables for a single OpenMP For or DO loop
// return the number of private variables added.
int collectLoopPrivateVariables(SgStatement *omp_loop) {
  int result = 0;
  ROSE_ASSERT(omp_loop != NULL);
  SgOmpForStatement *for_node = isSgOmpForStatement(omp_loop);
  SgOmpTargetParallelForStatement *target_parallel_for_node =
      isSgOmpTargetParallelForStatement(omp_loop);
  SgOmpDoStatement *do_node = isSgOmpDoStatement(omp_loop);
  if (for_node)
    omp_loop = for_node;
  else if (do_node)
    omp_loop = do_node;
  else if (target_parallel_for_node)
    omp_loop = target_parallel_for_node;
  else
    ROSE_ABORT();

  SgScopeStatement *directive_scope = omp_loop->get_scope();
  ROSE_ASSERT(directive_scope != NULL);
  // Collected nested loops and their indices
  // skip the top level loop?
  Rose_STL_Container<SgNode *> loops;
  if (for_node)
    loops = NodeQuery::querySubTree(for_node->get_body(), V_SgForStatement);
  else if (target_parallel_for_node)
    loops = NodeQuery::querySubTree(target_parallel_for_node->get_body(),
                                    V_SgForStatement);
  else if (do_node)
    loops = NodeQuery::querySubTree(do_node->get_body(), V_SgFortranDo);
  else
    ROSE_ABORT();
  // For all loops within the OpenMP loop
  Rose_STL_Container<SgNode *>::iterator loopIter = loops.begin();
  for (; loopIter != loops.end(); loopIter++) {
    SgInitializedName *index_var =
        SageInterface::getLoopIndexVariable(*loopIter);
    ROSE_ASSERT(index_var != NULL);
    SgScopeStatement *var_scope = index_var->get_scope();
    // Only loop index variables declared in higher  or the same scopes matter
    if (SageInterface::isAncestor(var_scope, directive_scope) ||
        var_scope == directive_scope) {
      // Grab possible enclosing parallel region
      bool isPrivateInRegion = false;
      SgOmpParallelStatement *omp_stmt = isSgOmpParallelStatement(
          SageInterface::getEnclosingNode<SgOmpParallelStatement>(omp_loop));
      if (omp_stmt) {
        isPrivateInRegion = OmpSupport::isInClauseVariableList(
            index_var, isSgOmpClauseBodyStatement(omp_stmt),
            V_SgOmpPrivateClause);
      }
      // add it into the private variable list only if it is not specified as
      // private in both the loop and region levels.
      if (!isPrivateInRegion &&
          !OmpSupport::isInClauseVariableList(
              index_var, isSgOmpClauseBodyStatement(omp_loop),
              V_SgOmpPrivateClause)) {
        result++;
        OmpSupport::addClauseVariable(index_var,
                                      isSgOmpClauseBodyStatement(omp_loop),
                                      V_SgOmpPrivateClause);
        SgVariableSymbol *data_symbol = isSgVariableSymbol(
            index_var->search_for_symbol_from_symbol_table());
        assert(data_symbol != NULL);
        ParallelData *data = new ParallelData(data_symbol);
        data->set_sharing_property("private");
        data->set_sharing_visibility("implicit");
        data->set_data_access("read-write");
        parallel_data[data_symbol] = data;
        std::cout << "Got a loop index to table: " << index_var->get_name()
                  << "\n";
      }
    }

  } // end for loops
  return result;
}

//! Patch up private variables for omp for. The reason is that loop indices
//! should be private by default and this function will make this explicit. This
//! should happen before the actual translation is done.
int collectPrivateVariables(SgFile *file) {
  int result = 0;
  ROSE_ASSERT(file != NULL);
  Rose_STL_Container<SgNode *> nodeList =
      NodeQuery::querySubTree(file, V_SgOmpForStatement);
  Rose_STL_Container<SgNode *> nodeList2 =
      NodeQuery::querySubTree(file, V_SgOmpDoStatement);

  Rose_STL_Container<SgNode *> nodeList2_merged(nodeList.size() +
                                                nodeList2.size());

  std::sort(nodeList.begin(), nodeList.end());
  std::sort(nodeList2.begin(), nodeList2.end());
  std::merge(nodeList.begin(), nodeList.end(), nodeList2.begin(),
             nodeList2.end(), nodeList2_merged.begin());

  // TODO: implement a helper function to collect all the nodes into one list
  Rose_STL_Container<SgNode *> nodeList3 =
      NodeQuery::querySubTree(file, V_SgOmpTargetParallelForStatement);
  std::sort(nodeList3.begin(), nodeList3.end());
  Rose_STL_Container<SgNode *> nodeList_merged;
  std::merge(nodeList2_merged.begin(), nodeList2_merged.end(),
             nodeList3.begin(), nodeList3.end(),
             std::insert_iterator<Rose_STL_Container<SgNode *>>(
                 nodeList_merged, nodeList_merged.end()));

  Rose_STL_Container<SgNode *>::iterator nodeListIterator =
      nodeList_merged.begin();
  // For each omp for/do statement
  for (; nodeListIterator != nodeList_merged.end(); ++nodeListIterator) {
    SgStatement *omp_loop = NULL;
    SgOmpForStatement *for_node = isSgOmpForStatement(*nodeListIterator);
    SgOmpDoStatement *do_node = isSgOmpDoStatement(*nodeListIterator);
    SgOmpTargetParallelForStatement *target_parallel_for_node =
        isSgOmpTargetParallelForStatement(*nodeListIterator);
    if (for_node)
      omp_loop = for_node;
    else if (do_node)
      omp_loop = do_node;
    else if (target_parallel_for_node)
      omp_loop = target_parallel_for_node;
    else
      ROSE_ASSERT(false);
    result += collectLoopPrivateVariables(omp_loop);
  } // end for omp for statments
  return result;
} // end patchUpPrivateVariables()

//! Collect threadprivate variables within the current project, return a set to
//! avoid duplicated elements
std::set<SgInitializedName *> collectThreadprivateVariables() {
  // Do the actual collection only once
  static bool calledOnce = false;
  static std::set<SgInitializedName *> result;

  if (calledOnce)
    return result;
  calledOnce = true;
  std::vector<SgOmpThreadprivateStatement *> tp_stmts =
      SageInterface::getSgNodeListFromMemoryPool<SgOmpThreadprivateStatement>();
  std::vector<SgOmpThreadprivateStatement *>::const_iterator c_iter;
  for (c_iter = tp_stmts.begin(); c_iter != tp_stmts.end(); c_iter++) {
    SgVarRefExpPtrList refs = (*c_iter)->get_variables();
    SgInitializedNamePtrList var_list; // = (*c_iter)->get_variables();
    for (size_t j = 0; j < refs.size(); j++)
      var_list.push_back(refs[j]->get_symbol()->get_declaration());
    std::copy(var_list.begin(), var_list.end(),
              std::inserter(result, result.end()));
  }
  return result;
}

// Check if a variable that is determined to be shared in all enclosing
// constructs, up to and including the innermost enclosing parallel construct,
// is shared start_stmt is the start point to find enclosing OpenMP constructs.
// It is excluded as an enclosing construct for itself.
// TODO: we only check if it is shared to the innermost enclosing parallel
// construct for now
static bool isSharedInEnclosingConstructs(SgInitializedName *init_var,
                                          SgStatement *start_stmt) {
  bool result = false;
  ROSE_ASSERT(init_var != NULL);
  ROSE_ASSERT(start_stmt != NULL);
  SgScopeStatement *var_scope = init_var->get_scope();
  //    SgScopeStatement* directive_scope = start_stmt->get_scope();
  // locally declared variables are private to the start_stmt
  // We should not do this here. It is irrelevant to this function.
  // if (isAncestor(start_stmt, init_var))
  //   return false;

  //   cout<<"Debug omp_lowering.cpp isSharedInEnclosingConstructs()
  //   SgInitializedName name = "<<init_var->get_name().getString()<<endl;
  SgOmpParallelStatement *enclosing_par_stmt =
      SageInterface::getEnclosingNode<SgOmpParallelStatement>(start_stmt,
                                                              false);
  // Lexically nested within a parallel region
  if (enclosing_par_stmt) {
    // locally declared variables are private to enclosing_par_stmt
    SgScopeStatement *enclosing_construct_scope =
        enclosing_par_stmt->get_scope();
    ROSE_ASSERT(enclosing_construct_scope != NULL);
    if (SageInterface::isAncestor(enclosing_construct_scope, var_scope))
      return false;

    // Explicitly declared as a shared variable
    if (OmpSupport::isInClauseVariableList(init_var, enclosing_par_stmt,
                                           V_SgOmpSharedClause))
      result = true;
    else { // shared by default
      VariantVector vv(V_SgOmpPrivateClause);
      vv.push_back(V_SgOmpFirstprivateClause);
      vv.push_back(V_SgOmpCopyinClause);
      vv.push_back(V_SgOmpReductionClause);
      if (OmpSupport::isInClauseVariableList(init_var, enclosing_par_stmt, vv))
        result = false;
      else
        result = true;
    }
  } else
  // the variable is in an orphaned construct
  // The variable could be
  // 1. a function parameter: it is private to its enclosing parallel region
  // 2. a global variable: either a threadprivate variable or shared by default
  // 3. is a variable declared within an orphaned function: it is private to its
  // enclosing parallel region
  // ?? any other cases?? TODO
  {
    SgFunctionDefinition *func_def =
        SageInterface::getEnclosingFunctionDefinition(start_stmt);
    ROSE_ASSERT(func_def != NULL);
    if (isSgGlobal(var_scope)) {
      std::set<SgInitializedName *> tp_vars = collectThreadprivateVariables();
      if (tp_vars.find(init_var) != tp_vars.end())
        result = false; // is threadprivate
      else
        result = true; // otherwise
    } else if (isSgFunctionParameterList(init_var->get_parent())) {
      // function parameters are private to its dynamically (non-lexically)
      // nested parallel regions.
      result = false;
    } else if (SageInterface::isAncestor(func_def, var_scope)) {
      // declared within an orphaned function, should be private
      result = false;
    } else {
      std::cerr
          << "Error: OmpSupport::isSharedInEnclosingConstructs() \n Unhandled "
             "variables within an orphaned construct:"
          << std::endl;
      std::cerr << "SgInitializedName name = "
                << init_var->get_name().getString() << std::endl;
      SageInterface::dumpInfo(init_var);
      init_var->get_file_info()->display("tttt");
      ROSE_ASSERT(false);
    }
  }
  return result;
} // end isSharedInEnclosingConstructs()

//! Patch up firstprivate variables for omp task. The reason is that the
//! specification 3.0 defines rules for implicitly determined data-sharing
//! attributes and this function will make the implicit firstprivate variable of
//! omp task explicit.
/*
variables used in task block:

2.9.1.1 Data-sharing Attribute Rules for Variables Referenced in a Construct
Ref. OMP 3.0 page 79
A variable is firstprivate to the task (default) , if
** not explicitly specified by default(), shared(),private(), firstprivate()
clauses
** not shared in enclosing constructs

It should also satisfy the restriction defined in specification 3.0 page 93 TODO
* cannot be a variable which is part of another variable (as an array or
structure element)
* cannot be private, reduction
* must have an accessible, unambiguous copy constructor for the class type
* must not have a const-qualified type unless it is of class type with a mutable
member
* must not have an incomplete C/C++ type or a reference type
*
I decided to exclude variables which are used by addresses when recognizing
firstprivate variables The reason is that in real code, it is often to have
private variables first then use their address later.   Skipping the replacement
will result in wrong semantics. e.g. from Allan Porterfield void    create_seq(
double seed, double a )
      {
             double x, s;
             int    i, k;

      #pragma omp parallel private(x,s,i,k)
         {
              // .....
             // here s is private
             s = find_my_seed( myid, num_procs,
                               (long)4*NUM_KEYS, seed, a );

             for (i=k1; i<k2; i++)
             {
                 x = randlc(&s, &a); // here s is used by its address

             }
         }
      }
If not, wrong code will be generated later on. The reason follows:
   * Considering nested omp tasks:
         #pragma omp task untied
            {
              int j =100;
              // i is firstprivate, item is shared
              {
                for (i = 0; i < LARGE_NUMBER; i++)
                {
      #pragma omp task if(1)
                  process (item[i],&j);
                }
              }
            }
   * the variable j will be firstprivate by default
   * however, it is used by its address within a nested task (&j)
   * replacing it with its local copy will not get the right, original address.
   *
   * Even worse: the replacement will cause some later translation (outlining)
to
   * access the address of a parent task's local variable.
   * It seems (not 100% certain!!!) that GOMP implements tasks as independent
entities.
   * As a result a parent task's local stack will not be always accessible to
its nested tasks.
   * A segmentation fault will occur when the lexically nested task tries to
obtain the address of
   * its parent task's local variable.
   * An example mistaken translation is shown below
       int main()
      {
        GOMP_parallel_start(OUT__3__1527__,0,0);
        OUT__3__1527__();
        GOMP_parallel_end();
        return 0;
      }

      void OUT__3__1527__()
      {
        if (GOMP_single_start()) {
          int i;
          printf(("Using %d threads.\n"),omp_get_num_threads());
          void *__out_argv2__1527__[1];
          __out_argv2__1527__[0] = ((void *)(&i));
          GOMP_task(OUT__2__1527__,&__out_argv2__1527__,0,4,4,1,1);
          //GOMP_task(OUT__2__1527__,&__out_argv2__1527__,0,4,4,1,0); //untied
or not, no difference
        }
      }

      void OUT__2__1527__(void **__out_argv)
{
  int *i = (int *)(__out_argv[0]);
  //  int _p_i;
  //  _p_i =  *i;
  //  for (_p_i = 0; _p_i < 1000; _p_i++) {
  for (*i = 0; *i < 1000; (*i)++) {
    void *__out_argv1__1527__[1];
    // cannot access auto variable from the stack of another task instance!!
    //__out_argv1__1527__[0] = ((void *)(&_p_i));
    __out_argv1__1527__[0] = ((void *)(&(*i)));// this is the right translation
    GOMP_task(OUT__1__1527__,&__out_argv1__1527__,0,4,4,1,0);
  }
}
void OUT__1__1527__(void **__out_argv)
{
  int *i = (int *)(__out_argv[0]);
  int _p_i;
  _p_i =  *i;
  assert(_p_i>=0);
  assert(_p_i<10000);

  process((item[_p_i]));
}
*
  */
int collectFirstprivateVariables(SgFile *file) {
  int result = 0;
  ROSE_ASSERT(file != NULL);
  Rose_STL_Container<SgNode *> nodeList =
      NodeQuery::querySubTree(file, V_SgOmpTaskStatement);
  Rose_STL_Container<SgNode *>::iterator iter = nodeList.begin();
  for (; iter != nodeList.end(); iter++) {
    SgOmpTaskStatement *target = isSgOmpTaskStatement(*iter);
    SgScopeStatement *directive_scope = target->get_scope();
    SgStatement *body = target->get_body();
    ROSE_ASSERT(body != NULL);

    // Find all variable references from the task's body
    Rose_STL_Container<SgNode *> refList =
        NodeQuery::querySubTree(body, V_SgVarRefExp);
    Rose_STL_Container<SgNode *>::iterator var_iter = refList.begin();
    for (; var_iter != refList.end(); var_iter++) {
      SgVarRefExp *var_ref = isSgVarRefExp(*var_iter);
      ROSE_ASSERT(var_ref->get_symbol() != NULL);
      SgInitializedName *init_var = var_ref->get_symbol()->get_declaration();
      ROSE_ASSERT(init_var != NULL);
      SgScopeStatement *var_scope = init_var->get_scope();
      ROSE_ASSERT(var_scope != NULL);

      // Variables with automatic storage duration that are declared in
      // a scope inside the construct are private. Skip them
      if (SageInterface::isAncestor(directive_scope, var_scope))
        continue;

      if (SageInterface::isUseByAddressVariableRef(var_ref))
        continue;
      // Skip variables already with explicit data-sharing attributes
      VariantVector vv(V_SgOmpDefaultClause);
      vv.push_back(V_SgOmpPrivateClause);
      vv.push_back(V_SgOmpSharedClause);
      vv.push_back(V_SgOmpFirstprivateClause);
      if (OmpSupport::isInClauseVariableList(init_var, target, vv))
        continue;
      // Skip variables which are class/structure members: part of another
      // variable
      if (isSgClassDefinition(init_var->get_scope()))
        continue;
      // Skip variables which are shared in enclosing constructs
      if (isSharedInEnclosingConstructs(init_var, target))
        continue;
      // Now it should be a firstprivate variable
      OmpSupport::addClauseVariable(init_var, target,
                                    V_SgOmpFirstprivateClause);
      result++;
    } // end for each variable reference
  }   // end for each SgOmpTaskStatement
  return result;
} // end patchUpFirstprivateVariables()

int collectImplicitMappingVariables(SgFile *file) {
  int result = 0;
  ROSE_ASSERT(file != NULL);
  Rose_STL_Container<SgNode *> node_list1 =
      NodeQuery::querySubTree(file, V_SgOmpParallelStatement);

  // TODO: implement a helper function to collect all the nodes into one list
  Rose_STL_Container<SgNode *> node_list2 =
      NodeQuery::querySubTree(file, V_SgOmpTargetParallelForStatement);
  std::sort(node_list1.begin(), node_list1.end());
  std::sort(node_list2.begin(), node_list2.end());
  Rose_STL_Container<SgNode *> node_list;
  std::merge(node_list1.begin(), node_list1.end(), node_list2.begin(),
             node_list2.end(),
             std::insert_iterator<Rose_STL_Container<SgNode *>>(
                 node_list, node_list.end()));

  Rose_STL_Container<SgNode *>::iterator iter = node_list.begin();
  for (iter = node_list.begin(); iter != node_list.end(); iter++) {
    SgOmpClauseBodyStatement *target = NULL;
    target = isSgOmpClauseBodyStatement(*iter);
    SgScopeStatement *directive_scope = target->get_scope();
    SgStatement *body = target->get_body();
    ROSE_ASSERT(body != NULL);

    // Find all variable references from the task's body
    Rose_STL_Container<SgNode *> ref_list =
        NodeQuery::querySubTree(body, V_SgVarRefExp);
    Rose_STL_Container<SgNode *>::iterator var_iter = ref_list.begin();
    for (var_iter = ref_list.begin(); var_iter != ref_list.end(); var_iter++) {
      SgVarRefExp *var_ref = isSgVarRefExp(*var_iter);

      SgNode *var_ref_parent = var_ref->get_parent();
      SgNode *var_ref_child = var_ref;
      while (isSgAssignOp(var_ref_parent) == NULL && var_ref_parent != NULL) {
        var_ref_child = var_ref_parent;
        var_ref_parent = var_ref_parent->get_parent();
      }

      ROSE_ASSERT(var_ref->get_symbol() != NULL);
      SgInitializedName *init_var = var_ref->get_symbol()->get_declaration();
      ROSE_ASSERT(init_var != NULL);
      SgScopeStatement *var_scope = init_var->get_scope();
      ROSE_ASSERT(var_scope != NULL);

      // Variables with automatic storage duration that are declared in
      // a scope inside the construct are private. Skip them
      if (SageInterface::isAncestor(directive_scope, var_scope))
        continue;

      // TODO: is it necessary?
      // if (SageInterface::isUseByAddressVariableRef(var_ref))
      // continue;

      // Skip variables already with explicit data-sharing attributes
      VariantVector vv(V_SgOmpDefaultClause);
      vv.push_back(V_SgOmpPrivateClause);
      vv.push_back(V_SgOmpSharedClause);
      vv.push_back(V_SgOmpFirstprivateClause);
      if (OmpSupport::isInClauseVariableList(init_var, target, vv))
        continue;
      // Skip variables which are class/structure members: part of another
      // variable
      if (isSgClassDefinition(init_var->get_scope()))
        continue;
      // Skip variables which are shared in enclosing constructs
      if (isSharedInEnclosingConstructs(init_var, target))
        continue;

      SgVariableSymbol *data_symbol =
          isSgVariableSymbol(init_var->search_for_symbol_from_symbol_table());
      assert(data_symbol != NULL);

      std::string data_access = "read-only";
      if (isSgAssignOp(var_ref_parent) &&
          isSgAssignOp(var_ref_parent)->get_lhs_operand() == var_ref_child) {
        std::cout << "Got a writing L value: "
                  << var_ref->get_symbol()->get_name() << "\n";
        data_access = "read-write";
      }

      // Now it should be a shared variable
      SgNode *parent = target->get_parent();
      if (isSgBasicBlock(parent)) // skip the padding block in between.
        parent = parent->get_parent();
      if (isSgOmpTargetStatement(parent) ||
          isSgOmpTargetParallelForStatement(target)) {
        SgVariableSymbol *sym = var_ref->get_symbol();
        ROSE_ASSERT(sym != NULL);
        SgType *orig_type = sym->get_type();
        SgArrayType *a_type = isSgArrayType(orig_type);
        SgOmpClauseBodyStatement *target_parent = NULL;
        if (isSgOmpTargetParallelForStatement(target)) {
          target_parent = isSgOmpTargetParallelForStatement(target);
        } else {
          target_parent = isSgOmpTargetStatement(parent);
        }
        if (a_type != NULL) {
          std::vector<SgExpression *> dims =
              SageInterface::get_C_array_dimensions(a_type);
          SgExpression *array_length = NULL;
          for (std::vector<SgExpression *>::const_iterator iter = dims.begin();
               iter != dims.end(); iter++) {
            SgExpression *length_exp = *iter;
            // TODO: get_C_array_dimensions returns one extra null expression
            // somehow.
            if (!isSgNullExpression(length_exp))
              array_length = length_exp;
          }
          ROSE_ASSERT(array_length != NULL);
          SgOmpMapClause *map_clause = NULL;
          std::map<SgSymbol *,
                   std::vector<std::pair<SgExpression *, SgExpression *>>>
              array_dimensions;
          SgExprListExp *explist = NULL;
          SgVariableSymbol *array_symbol = var_ref->get_symbol();

          if (OmpSupport::hasClause(target_parent, V_SgOmpMapClause)) {
            Rose_STL_Container<SgOmpClause *> map_clauses =
                OmpSupport::getClause(target_parent, V_SgOmpMapClause);
            Rose_STL_Container<SgOmpClause *>::const_iterator iter;
            for (iter = map_clauses.begin(); iter != map_clauses.end();
                 iter++) {
              SgOmpMapClause *temp_map_clause = isSgOmpMapClause(*iter);
              if (temp_map_clause->get_operation() ==
                  SgOmpClause::e_omp_map_tofrom) {
                map_clause = temp_map_clause;
                array_dimensions = map_clause->get_array_dimensions();
                explist = map_clause->get_variables();
                break;
              }
            }
          }

          if (map_clause == NULL) {
            explist = SageBuilder::buildExprListExp();
            SgOmpClause::omp_map_operator_enum sg_type =
                SgOmpClause::e_omp_map_tofrom;
            map_clause = new SgOmpMapClause(explist, sg_type);
            explist->set_parent(map_clause);
            SageInterface::setOneSourcePositionForTransformation(map_clause);
            map_clause->set_parent(target_parent);
          }

          bool has_mapped = false;
          Rose_STL_Container<SgExpression *>::iterator iter;
          SgExpressionPtrList expression_list = explist->get_expressions();
          for (iter = expression_list.begin(); iter != expression_list.end();
               iter++) {
            if (isSgVarRefExp(*iter)->get_symbol() == array_symbol) {
              has_mapped = true;
              break;
            }
          }

          if (has_mapped == false) {
            SgExpression *lower_exp = SageBuilder::buildIntVal(0);
            explist->append_expression(
                SageBuilder::buildVarRefExp(var_ref->get_symbol()));
            array_dimensions[array_symbol].push_back(
                std::make_pair(lower_exp, array_length));
            map_clause->set_array_dimensions(array_dimensions);
          }

        } else {
          OmpSupport::addClauseVariable(init_var, target_parent,
                                        V_SgOmpFirstprivateClause);
          if (parallel_data.count(data_symbol) == 0) {
            ParallelData *data = new ParallelData(data_symbol);
            data->set_sharing_property("firstprivate");
            data->set_sharing_visibility("implicit");
            data->set_data_access(data_access);
            parallel_data[data_symbol] = data;
            std::cout << "Got an implicit firstprivate var to table: "
                      << init_var->get_name() << "\n";
            continue;
          } else {
            if (data_access == "read-write") {
              parallel_data[data_symbol]->set_data_access(data_access);
            }
          }
        }
      } else {
        OmpSupport::addClauseVariable(init_var, target, V_SgOmpSharedClause);
          if (parallel_data.count(data_symbol) == 0) {
            ParallelData *data = new ParallelData(data_symbol);
            data->set_sharing_property("shared");
            data->set_sharing_visibility("implicit");
            data->set_data_access(data_access);
            parallel_data[data_symbol] = data;
            std::cout << "Got an implicit shared var to table: "
                      << init_var->get_name() << "\n";
            continue;
          } else {
            if (data_access == "read-write") {
              parallel_data[data_symbol]->set_data_access(data_access);
            }
          }
      }
      result++;
    } // end for each variable reference
  }
  return result;
} // end patchUpImplicitMappingVariables()

int collectExplicitMappingVariables(SgFile *file) {
  int result = 0;
  ROSE_ASSERT(file != NULL);
  Rose_STL_Container<SgNode *> node_list1 =
      NodeQuery::querySubTree(file, V_SgOmpParallelStatement);

  // TODO: implement a helper function to collect all the nodes into one list
  Rose_STL_Container<SgNode *> node_list2 =
      NodeQuery::querySubTree(file, V_SgOmpTargetParallelForStatement);
  std::sort(node_list1.begin(), node_list1.end());
  std::sort(node_list2.begin(), node_list2.end());
  Rose_STL_Container<SgNode *> node_list;
  std::merge(node_list1.begin(), node_list1.end(), node_list2.begin(),
             node_list2.end(),
             std::insert_iterator<Rose_STL_Container<SgNode *>>(
                 node_list, node_list.end()));

  Rose_STL_Container<SgNode *>::iterator iter = node_list.begin();
  for (iter = node_list.begin(); iter != node_list.end(); iter++) {
    SgOmpClauseBodyStatement *target = NULL;
    target = isSgOmpClauseBodyStatement(*iter);
    SgScopeStatement *directive_scope = target->get_scope();
    SgStatement *body = target->get_body();
    ROSE_ASSERT(body != NULL);

    // Find all variable references from the task's body
    Rose_STL_Container<SgNode *> ref_list =
        NodeQuery::querySubTree(body, V_SgVarRefExp);
    Rose_STL_Container<SgNode *>::iterator var_iter = ref_list.begin();
    for (var_iter = ref_list.begin(); var_iter != ref_list.end(); var_iter++) {
      SgVarRefExp *var_ref = isSgVarRefExp(*var_iter);

      SgNode *var_ref_parent = var_ref->get_parent();
      SgNode *var_ref_child = var_ref;
      while (isSgAssignOp(var_ref_parent) == NULL && var_ref_parent != NULL) {
        var_ref_child = var_ref_parent;
        var_ref_parent = var_ref_parent->get_parent();
      }

      std::string data_access = "read-only";
      if (isSgAssignOp(var_ref_parent) &&
          isSgAssignOp(var_ref_parent)->get_lhs_operand() == var_ref_child) {
        std::cout << "Got a writing L value: "
                  << var_ref->get_symbol()->get_name() << "\n";
        data_access = "read-write";
      }

      ROSE_ASSERT(var_ref->get_symbol() != NULL);
      SgInitializedName *init_var = var_ref->get_symbol()->get_declaration();
      ROSE_ASSERT(init_var != NULL);
      SgScopeStatement *var_scope = init_var->get_scope();
      ROSE_ASSERT(var_scope != NULL);

      // Variables with automatic storage duration that are declared in
      // a scope inside the construct are private. Skip them
      if (SageInterface::isAncestor(directive_scope, var_scope))
        continue;

      // TODO: is it necessary?
      // if (SageInterface::isUseByAddressVariableRef(var_ref))
      // continue;

      SgVariableSymbol *data_symbol =
          isSgVariableSymbol(init_var->search_for_symbol_from_symbol_table());
      assert(data_symbol != NULL);
      // Skip variables already with explicit data-sharing attributes
      VariantVector vv(V_SgOmpPrivateClause);
      if (OmpSupport::isInClauseVariableList(init_var, target, vv)) {
        if (parallel_data.count(data_symbol) == 0) {
          ParallelData *data = new ParallelData(data_symbol);
          data->set_sharing_property("private");
          data->set_sharing_visibility("explicit");
          data->set_data_access(data_access);
          parallel_data[data_symbol] = data;
          std::cout << "Got a private var to table: " << init_var->get_name()
                    << "\n";
          continue;
        } else {
          if (data_access == "read-write") {
            parallel_data[data_symbol]->set_data_access(data_access);
          }
        }
      }
      vv.pop_back();
      vv.push_back(V_SgOmpSharedClause);
      if (OmpSupport::isInClauseVariableList(init_var, target, vv)) {
        if (parallel_data.count(data_symbol) == 0) {
          ParallelData *data = new ParallelData(data_symbol);
          data->set_sharing_property("shared");
          data->set_sharing_visibility("explicit");
          data->set_data_access(data_access);
          parallel_data[data_symbol] = data;
          std::cout << "Got a shared var to table: " << init_var->get_name()
                    << "\n";
          continue;
        } else {
          if (data_access == "read-write") {
            parallel_data[data_symbol]->set_data_access(data_access);
          }
        }
      }
      vv.pop_back();
      vv.push_back(V_SgOmpFirstprivateClause);
      if (OmpSupport::isInClauseVariableList(init_var, target, vv)) {
        if (parallel_data.count(data_symbol) == 0) {
          ParallelData *data = new ParallelData(data_symbol);
          data->set_sharing_property("firstprivate");
          data->set_sharing_visibility("explicit");
          data->set_data_access(data_access);
          parallel_data[data_symbol] = data;
          std::cout << "Got a firstprivate var to table: "
                    << init_var->get_name() << "\n";
          continue;
        } else {
          if (data_access == "read-write") {
            parallel_data[data_symbol]->set_data_access(data_access);
          }
        }
      }
      // Skip variables which are class/structure members: part of another
      // variable
      if (isSgClassDefinition(init_var->get_scope()))
        continue;
      // Skip variables which are shared in enclosing constructs
      if (isSharedInEnclosingConstructs(init_var, target))
        continue;
      result++;
    } // end for each variable reference
  }
  return result;
} // end patchUpExplicitMappingVariables()

std::map<SgVariableSymbol *, ParallelData *> analyze_parallel_data(SgSourceFile *file) {

  collectExplicitMappingVariables(file);

  collectPrivateVariables(file); // the order of these two functions matter! We
                                 // want to patch up private variable first!
  collectFirstprivateVariables(file);

  collectImplicitMappingVariables(file);

  return parallel_data;
}
