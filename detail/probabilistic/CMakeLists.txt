cmake_minimum_required( VERSION 3.13 )
cmake_policy(SET CMP0077 NEW)

project( probabilistic )
macro( project )
    if ( NOT "${ARGV0}" STREQUAL "probabilistic" )
        _project(${ARGV})
    endif()
endmacro()

# Standard debug flags causes the assembler to
# fail for being too large.
# https://stackoverflow.com/questions/14125007/gcc-string-table-overflow-error-during-compilation#29479701
if ( WIN32 )
    set( CMAKE_CXX_FLAGS_DEBUG "-O2 -g" )
endif()

include( "probabilistic/src/build_type.cmake") 

# If we want to be able to build without relying on R CMD INSTALL
# (for example to build tests outside of R context) then we need
# to pass in the directory with the R libraries.
if ( R_BIN_DIR )
    link_directories( "${R_BIN_DIR}" )
endif()

# The Torch package must be found in the top level scope, see
# https://github.com/pytorch/pytorch/issues/21737.
# This file is not run when installing the R package, so we can check for the
# existance of this variable to detect whether code is being compiled by
# R CMD INSTALL or whether it is being compiled for testing of C++ code.
set(NO_R_PACKAGE TRUE)
#include("${CMAKE_CURRENT_SOURCE_DIR}/probabilistic/src/pytorch.cmake")

add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/probabilistic/src" )

add_subdirectory( "${CMAKE_CURRENT_SOURCE_DIR}/test" )

