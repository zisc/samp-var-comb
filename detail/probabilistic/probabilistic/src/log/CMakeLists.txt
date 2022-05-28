cmake_minimum_required( VERSION 3.0 )

add_library( log INTERFACE )
target_link_libraries( log
    INTERFACE boost
)
target_include_directories( log
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

