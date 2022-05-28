cmake_minimum_required(VERSION 3.0)

add_library(std_specialisations INTERFACE)
target_include_directories(std_specialisations
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

