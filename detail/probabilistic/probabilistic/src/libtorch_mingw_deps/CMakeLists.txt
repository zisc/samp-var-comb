cmake_minimum_required(VERSION 3.0)

if (MINGW)
    add_library(libtorch_mingw_deps STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/assert_fail.c"
    )
    target_include_directories(libtorch_mingw_deps
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()
