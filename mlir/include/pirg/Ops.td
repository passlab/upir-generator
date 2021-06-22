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

def SpmdOp : Pirg_Op<"spmd", [AttrSizedOperandSegments]> {
  let summary = "spmd operation";
  let description = [{
    Create a SPMD region and perform parallel computing.
  }];

  let regions = (region AnyRegion:$region);

  let arguments = (ins Optional<AnyType>:$num_units,
             Optional<AnyType>:$nested_parallelism_level,
             Variadic<AnyType>:$nested_parallelism_list,
             Variadic<AnyType>:$data,
             Optional<AnyType>:$cost);

  let builders = [
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let printer = [{ return ::printSpmdOp(p, *this); }];

}

def WorkshareOp : Pirg_Op<"workshare", [AttrSizedOperandSegments, AllTypesMatch<["lower_bound", "upper_bound", "step"]>]> {
  let summary = "worksharing operation";
  let description = [{
    Distribute workload among the threads in a parallel region.
  }];

  let regions = (region AnyRegion:$region);

  let arguments = (ins Optional<AnyType>:$chunk_size,
             Variadic<Index>:$lower_bound,
             Variadic<Index>:$upper_bound,
             Variadic<Index>:$step,
             OptionalAttr<StrAttr>:$schedule_policy,
             Optional<AnyType>:$collapse,
             Variadic<AnyType>:$distribute,
             Optional<AnyType>:$simdlen,
             Variadic<AnyType>:$data,
             Optional<AnyType>:$cost);

  let builders = [
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

}

def TaskOp : Pirg_Op<"task", [AttrSizedOperandSegments]> {
  let summary = "task operation";
  let description = [{
    Create a task region, which can be handled on the host or devices.
  }];

  let regions = (region AnyRegion:$region);

  let arguments = (ins Optional<AnyType>:$device_id,
             OptionalAttr<StrAttr>:$device,
             Optional<AnyType>:$condition,
             Variadic<AnyType>:$depend,
             Optional<AnyType>:$cost);

  let builders = [
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let printer = [{ return ::printTaskOp(p, *this); }];

}

def DataOp : Pirg_Op<"data", [AttrSizedOperandSegments]> {
  let summary = "data operation";
  let description = [{
    Create a data environment and/or perform data movements.
  }];

  let regions = (region AnyRegion:$region);

  let arguments = (ins Optional<AnyType>:$device_id,
             OptionalAttr<StrAttr>:$device,
             Variadic<AnyType>:$enter_vars,
             Variadic<AnyType>:$update_vars,
             Optional<AnyType>:$exit,
             Optional<AnyType>:$cost);

  let builders = [
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let printer = [{ return ::printDataOp(p, *this); }];

}

