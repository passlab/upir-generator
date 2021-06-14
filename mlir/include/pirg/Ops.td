//===- Ops.td - Pirg dialect operation definitions ---------*- tablegen -*-===//
//
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Pirg dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'pirg' dialect in the ODS framework so that we
// can define our operations.
def Pirg_Dialect : Dialect {
  let name = "pirg";
  let cppNamespace = "::mlir::pirg";
}

// Base class for pirg dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Pirg_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Pirg_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Pirg Operations
//===----------------------------------------------------------------------===//

// We define a pirg operation by inheriting from our base 'Pirg_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation.
// An operation could be marked as 'NoSideEffect' as it is a pure operation
// and may be removed if dead.

def SpmdOp : Pirg_Op<"spmd"> {
  let summary = "spmd operation";
  let description = [{
    Create a SPMD region and perform parallel computing.
  }];

  let regions = (region AnyRegion:$region);

  let arguments = (ins Optional<AnyType>:$num_threads_var);

  let builders = [
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let printer = [{ return ::printSpmdOp(p, *this); }];

}

