//===- Dialect.cpp - Upir IR Dialect registration in MLIR -----------------===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements the Upir dialect
//
//===----------------------------------------------------------------------===//

#include "upir/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::upir;

//===----------------------------------------------------------------------===//
// Upir Operations
//===----------------------------------------------------------------------===//

void SpmdOp::print(mlir::OpAsmPrinter &printer) {
  if (auto threads = this->num_units())
    printer << " num_units(" << threads << " : " << threads.getType() << ")";

  mlir::OperandRange data = this->data();
  if (data.size()) {
    printer << " data(";
    unsigned int i;
    for (i = 0; i < data.size(); i++) {
      if (i != 0) {
        printer << ", ";
      }
      printer << data[i];
    }
    printer << ")";
  }
  printer << " ";

  printer.printRegion(this->getRegion());
}

mlir::ParseResult SpmdOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  return mlir::success();
}

void WorkshareOp::print(mlir::OpAsmPrinter &printer) {
  if (auto collapse = this->collapse())
    printer << " collapse(" << collapse << " : " << collapse.getType() << ")";

  printer << " ";
  printer.printRegion(this->getRegion());
}

mlir::ParseResult WorkshareOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  return mlir::success();
}

void TaskOp::print(mlir::OpAsmPrinter &printer) {
  if (auto device = this->device()) {
    printer << " target(" << device;
    if (auto device_id = this->device_id()) {
      printer << " : " << device_id;
    }
    printer << ")";
  }

  mlir::OperandRange data = this->data();
  if (data.size()) {
    printer << " data(";
    unsigned int i;
    for (i = 0; i < data.size(); i++) {
      if (i != 0) {
        printer << ", ";
      }
      printer << data[i];
    }
    printer << ")";
  }
  printer << " ";
  printer.printRegion(this->getRegion());
}

mlir::ParseResult TaskOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  return mlir::success();
}

void DataOp::print(mlir::OpAsmPrinter &printer) {
  if (auto device = this->device()) {
    printer << " target(" << device;
    if (auto device_id = this->device_id()) {
      printer << " : " << device_id;
    }
    printer << ")";
  }
  printer << " ";
  printer.printRegion(this->getRegion());
}

mlir::ParseResult DataOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  return mlir::success();
}

void BarrierOp::print(mlir::OpAsmPrinter &printer) {
  if (auto task = this->task_id()) {
    printer << " task(" << task << ")";
  }
  if (auto implicit = this->implicit()) {
    printer << " implicit";
  }
  printer << " ";
}

mlir::ParseResult BarrierOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  return mlir::success();
}

void ParallelDataInfoOp::print(mlir::OpAsmPrinter &printer) {
  printer << " (";
  mlir::ArrayAttr array_attr = this->data();
  llvm::ArrayRef<mlir::Attribute> string_attr_list = array_attr.getValue();
  llvm::ArrayRef<mlir::Attribute>::iterator iter;
  for (iter = string_attr_list.begin(); iter != string_attr_list.end();
       iter++) {
    if (iter != string_attr_list.begin()) {
      printer << ", ";
    }
    const mlir::StringAttr *string_attr =
        static_cast<const mlir::StringAttr *>(iter);
    llvm::StringRef s = string_attr->getValue();
    if (s == "") {
      s = "n/a";
    }
    printer << s;
  }
  if (auto ssa_id = this->value()) {
    printer << " : " << ssa_id;
  }

  printer << ")";
  printer << " ";
}

mlir::ParseResult ParallelDataInfoOp::parse(mlir::OpAsmParser &parser,
                                            mlir::OperationState &result) {
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace upir {
namespace detail {
/// This class represents the internal storage of the Toy `StructType`.
struct ParallelDataTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::StringAttr>;

  /// A constructor for the type storage instance.
  ParallelDataTypeStorage(llvm::ArrayRef<mlir::StringAttr> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::StringAttr> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static ParallelDataTypeStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::StringAttr> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<ParallelDataTypeStorage>())
        ParallelDataTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::StringAttr> elementTypes;
};
} // end namespace detail
} // namespace upir
} // end namespace mlir

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
ParallelDataType
ParallelDataType::get(llvm::ArrayRef<mlir::StringAttr> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after the context are forwarded to the storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::StringAttr> ParallelDataType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}

/// Parse an instance of a type registered to the toy dialect.
mlir::Type UpirDialect::parseType(mlir::DialectAsmParser &parser) const {
  // TODO
  return nullptr;
}

/// Print an instance of a type registered to the toy dialect.
void UpirDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  // ParallelDataType parallelDataType = type.cast<ParallelDataType>();

  // Print the struct type according to the parser format.
  // printer << "struct<";
  // llvm::interleaveComma(parallelDataType.getElementTypes(), printer);
  // printer << '>';
}

#include "upir/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "upir/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// UpirDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void UpirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "upir/Ops.cpp.inc"
      >();
  addTypes<ParallelDataType>();
}
