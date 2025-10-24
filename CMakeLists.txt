cmake_minimum_required(VERSION 4.1)
project(dynsys LANGUAGES CXX C)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL /arch:AVX2 /fp:fast")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -flto -funroll-loops -fomit-frame-pointer -fPIC")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
    endif()
endif()

add_subdirectory(backend)
add_subdirectory(external/clustering)

# find_package(hip REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

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
target_link_libraries(dynsys PRIVATE dynsysBackend LLVM-21 clang-cpp clustering_header_lib OpenMP::OpenMP_CXX)


set_target_properties(dynsys PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dynsys
  PREFIX "_" )
