cmake_minimum_required( VERSION 3.0 )

add_library(Faddeeva STATIC "${CMAKE_CURRENT_SOURCE_DIR}/Faddeeva/src/Faddeeva.cc")
target_include_directories(Faddeeva PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Faddeeva/include")
set_target_properties(Faddeeva PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

