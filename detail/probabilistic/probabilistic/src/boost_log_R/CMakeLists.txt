cmake_minimum_required( VERSION 3.0 )

set( boost_log_R_src "${CMAKE_CURRENT_SOURCE_DIR}/src" )
add_library( boost_log_R STATIC
    "${boost_log_R_src}/filter.cpp"
    "${boost_log_R_src}/log_exception.cpp"
    "${boost_log_R_src}/sink_backend.cpp"
    "${boost_log_R_src}/string_to_severity_level.cpp"
)
target_link_libraries( boost_log_R
    PUBLIC boost
    PUBLIC Rlib
)
target_include_directories( boost_log_R
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_definitions( boost_log_R
    PUBLIC R_USE_C99_IN_CXX
)
if ( PROBABILISTIC_SILENCE_WARNS )
    target_compile_options( boost_log_R
        PRIVATE -Wno-deprecated-declarations
    )
endif()

