cmake_minimum_required(VERSION 4.1)
project(dynsys LANGUAGES CXX C)

add_subdirectory(backend)

# find_package(hip REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(pybind11 REQUIRED)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

pybind11_add_module(dynsys src/bindings.cpp src/jit.cpp src/simulation.cpp)
target_include_directories(dynsys PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(dynsys PRIVATE -Wall -Wextra )

message(STATUS "Using LLVMConfig.cmake: ${LLVM_CONFIG}")
message(STATUS "LLVM version: ${LLVM_VERSION}")
message(STATUS "Using ClangConfig.cmake: ${Clang_CONFIG}")
message(STATUS "Clang version: ${Clang_VERSION}")
set_target_properties(dynsys PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
target_link_libraries(dynsys PRIVATE dynsysBackend LLVM-20 clang-cpp)


set_target_properties(dynsys PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dynsys
  PREFIX "_" )
