project(dynsysBackend LANGUAGES CXX)

add_library(dynsysBackend 
    src/cpu/integrator.cpp
)

target_include_directories(dynsysBackend 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

