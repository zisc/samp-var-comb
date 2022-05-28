# Partially retrieved from
# https://blog.kitware.com/cmake-and-the-default-build-type/

# Set a default build type if none was specified.

set( default_build_type "Release" )
if(EXISTS "${CMAKE_SOURCE_DIR}/.git" OR EXISTS "${CMAKE_SOURCE_DIR}/.gitignore")
  set(default_build_type "Debug")
endif()

if (DEFINED ENV{BUILD})
    set(CMAKE_BUILD_TYPE "$ENV{BUILD}" CACHE STRING "Choose the type of build." FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

