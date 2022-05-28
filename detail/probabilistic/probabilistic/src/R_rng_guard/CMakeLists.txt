cmake_minimum_required( VERSION 3.0 )

add_library( R_rng_guard INTERFACE )
target_include_directories( R_rng_guard
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

