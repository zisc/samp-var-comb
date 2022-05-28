cmake_minimum_required(VERSION 3.0)

add_library(R_support STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/function.cpp"
)
target_include_directories(R_support
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(R_support
    PUBLIC R_protect_guard
    PUBLIC boost_log_R
)

