cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
cmake_policy(SET CMP0077 NEW)
# See https://cmake.org/cmake/help/latest/policy/CMP0077.html#policy:CMP0077

set(PYTHON_EXECUTABLE "python3" CACHE STRING "Python Executable.")
set(ATEN_NO_TEST ON CACHE BOOL "Aten no test.")
set(BUILD_PYTHON OFF CACHE BOOL "Build python.")
set(USE_CUDA OFF CACHE BOOL "Use cuda.")
set(USE_ROCM OFF CACHE BOOL "Use rocm.")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs.")

# To compile pytorch as a Release build and probabilistic as a Debug build,
# we need to clear the debug postfix for the protobuf library, c.f.
# https://cmake.org/cmake/help/latest/prop_tgt/CONFIG_POSTFIX.html. Otherwise,
# cmake compiles the protobuf target without the postfix, but applies the postfix
# when linking protobuf as part of the target probabilistic, causing an error.
set(protobuf_DEBUG_POSTFIX "" CACHE STRING "Default debug postfix")

# The current version of the library mkl-dnn used by pytorch doesn't compile
# with gcc version 11, which is the version of gcc used by the R docker
# image. Disabling for now.
set(USE_MKLDNN OFF CACHE BOOL "Use MKLDNN. Only available on x86, x86_64, and AArch64.")

# gcc version 11 now treats nonnull warnings as errors, and we need to
# revert this behaviour to compile version 1.10.1 of pytorch.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-error=nonnull)
endif()

if (PROBABILISTIC_SILENCE_WARNS)
    add_compile_options(
        -Wno-deprecated-declarations
        -Wno-nonnull
        -Wno-stringop-overflow
        -Wno-unused-result
    )
endif()

set(save_probabilistic_source_dir "${probabilistic_SOURCE_DIR}")
set(new_probabilistic_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")

set(save_cmake_binary_dir "${CMAKE_BINARY_DIR}")
set(new_cmake_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")

set(save_cmake_build_type "${CMAKE_BUILD_TYPE}")
set(new_cmake_build_type "Release")

set(GENERATING_PYTORCH_BUILD TRUE)
set(GENERATING_PYTORCH_BUILD TRUE PARENT_SCOPE)

macro(old_env)
    set(probabilistic_SOURCE_DIR "${save_probabilistic_source_dir}")
    set(CMAKE_BINARY_DIR "${save_cmake_binary_dir}")
    set(CMAKE_BUILD_TYPE "${save_cmake_build_type}")
endmacro()

macro(new_env)
    if (GENERATING_PYTORCH_BUILD)
        set(probabilistic_SOURCE_DIR "${new_probabilistic_source_dir}")
        set(CMAKE_BINARY_DIR "${new_cmake_binary_dir}")
        set(CMAKE_BUILD_TYPE "${new_cmake_build_type}")
    endif()
endmacro()

macro(cmake_minimum_required)
    new_env()
    _cmake_minimum_required(${ARGV})
endmacro()

macro(include)
    new_env()
    _include(${ARGV})
endmacro()

macro(target_include_directories)
    new_env()
    _target_include_directories(${ARGV})
endmacro()

macro(add_custom_command)
    new_env()
    _add_custom_command(${ARGV})
endmacro()

new_env()

