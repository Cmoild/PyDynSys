cmake_minimum_required(VERSION 4.1)
project(dynsys LANGUAGES CXX C)

add_subdirectory(backend)
add_subdirectory(external/clustering)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

if (DEFINED LLVM_INCLUDE_DIRS)
  include_directories(${LLVM_INCLUDE_DIRS})
endif()
if (DEFINED LLVM_DEFINITIONS)
  add_definitions(${LLVM_DEFINITIONS})
endif()

message(STATUS "Using LLVMConfig.cmake: ${LLVM_CONFIG}")
message(STATUS "LLVM version: ${LLVM_VERSION}")
message(STATUS "Using ClangConfig.cmake: ${Clang_CONFIG}")
message(STATUS "Clang version: ${Clang_VERSION}")

pybind11_add_module(dynsys src/bindings.cpp src/jit.cpp src/simulation.cpp)
target_include_directories(dynsys PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (MSVC)
  target_compile_options(dynsys PRIVATE /W3)
else()
  target_compile_options(dynsys PRIVATE -Wall -Wextra)
endif()

if (MSVC)
  target_compile_options(dynsys PRIVATE
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:Release>:/GL>
    $<$<CONFIG:Release>:/arch:AVX2>
    $<$<CONFIG:Release>:/fp:fast>
  )
  target_link_options(dynsys PRIVATE
    $<$<CONFIG:Release>:/LTCG>
  )
else()
  target_compile_options(dynsys PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-mtune=native>
    $<$<CONFIG:Release>:-flto>
    $<$<CONFIG:Release>:-funroll-loops>
    $<$<CONFIG:Release>:-fomit-frame-pointer>
    $<$<CONFIG:Release>:-fPIC>
  )
  target_link_options(dynsys PRIVATE
    $<$<CONFIG:Release>:-flto>
  )
endif()

set_target_properties(dynsys PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

if (TARGET LLVM::LLVM)
  target_link_libraries(dynsys PRIVATE dynsysBackend LLVM::LLVM clang-cpp clustering_header_lib OpenMP::OpenMP_CXX)
else()
  target_link_libraries(dynsys PRIVATE dynsysBackend LLVM-21 clang-cpp clustering_header_lib OpenMP::OpenMP_CXX)
endif()

set(OUTPUT_DIR ${CMAKE_SOURCE_DIR}/dynsys)
set_target_properties(dynsys PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
)

if (WIN32)
  set_target_properties(dynsys PROPERTIES PREFIX "_" SUFFIX ".pyd")
  set_property(GLOBAL PROPERTY CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  set_target_properties(dynsys PROPERTIES PREFIX "_" )
endif()

message(STATUS "Library output directory: ${OUTPUT_DIR}")
if (WIN32)
  message(STATUS "Windows-specific build: setting module suffix to .pyd and PREFIX to empty")
endif()

