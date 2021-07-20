cmake_minimum_required(VERSION 3.13.4)
project(standalone-dialect LANGUAGES CXX C)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${REX_INSTALL}/include/rose)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${REX_INSTALL}/lib)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(standalone-opt)
add_subdirectory(standalone-translate)

set(LLVM_LINK_COMPONENTS
  Support
  )

# rex2mlir source files
set(REX2MLIR_SOURCE_FILES
    rex2mlir.cpp
    mlir/MLIRGen.cpp
    mlir/Dialect.cpp
    analysis/data_analyzing.cpp
)

add_executable(rex2mlir ${REX2MLIR_SOURCE_FILES})
add_dependencies(rex2mlir PirgOpsIncGen)
target_compile_options(rex2mlir PUBLIC -Wno-misleading-indentation)
target_link_libraries(rex2mlir
  PRIVATE
    rose
    boost_system
    MLIRAnalysis
    MLIRIR
    MLIROpenMP
    MLIRLLVMIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)
