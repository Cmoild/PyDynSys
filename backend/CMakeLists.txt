project(dynsysBackend LANGUAGES CXX ASM)

add_library(dynsysBackend 
    src/cpu/integrator.cpp
    src/cpu/lu_chen.S
)

target_include_directories(dynsysBackend 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

