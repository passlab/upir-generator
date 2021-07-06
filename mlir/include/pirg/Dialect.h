//===- Dialect.h - Dialect definition for the Pirg IR ----------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR of Pirg Dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace pirg {
namespace detail {
struct ParallelDataTypeStorage;
} // end namespace detail
} // end namespace toy
} // end namespace mlir

/// Include the auto-generated header file containing the declaration of the
/// pirg dialect.
#include "pirg/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// pirg operations.
#define GET_OP_CLASSES
#include "pirg/Ops.h.inc"

namespace mlir {
namespace pirg {

//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class ParallelDataType : public mlir::Type::TypeBase<ParallelDataType, mlir::Type,
                                               detail::ParallelDataTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static ParallelDataType get(llvm::ArrayRef<mlir::StringAttr> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::StringAttr> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
} // end namespace toy
} // end namespace mlir

