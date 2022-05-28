cmake_minimum_required( VERSION 3.13 )
cmake_policy(SET CMP0077 NEW)

# See "Package Manager" section of Effective CMake
# conference talk.
# Video:
# https://www.youtube.com/watch?v=bsXLMQ6WgIk
# Slides:
# https://github.com/boostcon/cppnow_presentations_2017/blob/master/05-19-2017_friday/effective_cmake__daniel_pfeifer__cppnow_05-19-2017.pdf

project( probabilistic )
#macro( project )
#    if ( NOT "${ARGV0}" STREQUAL "probabilistic" )
#        _project(${ARGV})
#    endif()
#endmacro()

# Noop install calls by third party CMakeLists.
# macro( install )
#     message( STATUS "install(${ARGV}) suppressed." )
# endmacro()

# Noop export calls by third party CMakeLists.
# macro( export )
#     message( STATUS "export(${ARGV}) suppressed." )
# endmacro()

# Use C11. GNU extensions needed for asm function.
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11" )

# Some of the variables CMAKE uses to reference programs
# (compilers, linkers, etc) are prepended by the working
# directory of cmake before being set to the RHS string.
# To set these variables properly, we first need to
# test whether they point to a file, and if not,
# reset them to the result of a find_program call.

# Find ar from R Makeconf AR variable.
set( BACKUP_CMAKE_AR "${CMAKE_AR}" )
if ( AR )
    set( CMAKE_AR "${AR}" )
    if ( NOT EXISTS "${CMAKE_AR}" )
        find_program( AR_CACHE "${AR}" )
        if ( "${AR_CACHE}" STREQUAL "AR_CACHE-NOTFOUND" )
            message( WARNING "ar command \"${AR}\" not found, reverting to CMake default ar." )
            set( CMAKE_AR "${BACKUP_CMAKE_AR}" )
        else()
            set( CMAKE_AR "${AR_CACHE}" )
        endif()
    endif()
else()
    message( STATUS "AR variable undefined, reverting to CMake default ar." )
endif()

# Find ranlib from R Makeconf RANLIB variable.
set( BACKUP_CMAKE_RANLIB "${CMAKE_RANLIB}" )
if ( RANLIB )
    set( CMAKE_RANLIB "${RANLIB}" )
    if ( NOT EXISTS "${CMAKE_RANLIB}" )
        find_program( RANLIB_CACHE "${RANLIB}" )
        if ( "${RANLIB_CACHE}" STREQUAL "RANLIB_CACHE-NOTFOUND" )
            message( STATUS "ranlib command \"${RANLIB}\" not found, reverting to CMake default." )
            set( CMAKE_RANLIB "${BACKUP_CMAKE_RANLIB}" )
        else()
            set( CMAKE_RANLIB "${RANLIB_CACHE}" )
        endif()
    endif()
else()
    message( STATUS "RANLIB variable undefined, reverting to CMake default ranlib." )
endif()

# In release build, use Rs Makeconf flags, and define/undefine (n)debug macros.
# However, don't compile debug symbols into the build.
if ( CFLAGS )
    set( CMAKE_C_FLAGS_RELEASE "${CFLAGS} -g0 -DNDEBUG -UDEBUG" )
else()
    message( STATUS "CFLAGS variable undefined, reverting to CMake defaults." )
endif()

if ( CXX11FLAGS )
    set( CMAKE_CXX_FLAGS_RELEASE "${CXX11FLAGS} -g0 -DNDEBUG -UDEBUG" )
else()
    message( STATUS "CXX11FLAGS variable undefined, reverting to CMake defaults." )
endif()

# Import shared library extension from Rs Makeconf.
if ( SHLIB_EXT )
    set( CMAKE_SHARED_LIBRARY_SUFFIX "${SHLIB_EXT}" )
else()
    message( STATUS "SHLIB_EXT variable undefined, reverting to CMake default." )
endif()

# Import base linker flags from Rs Makeconf
if ( LDFLAGS )
    set( CMAKE_SHARED_LINKER_FLAGS "${LDFLAGS}" )
else()
    message( STATUS "LDFLAGS variable undefined, reverting to CMake defaults." )
endif()

# Import module linker flags from Rs Makeconf.
if ( SHLIB_CXX11LDFLAGS )
    set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SHLIB_CXX11LDFLAGS}" )
else()
    message( STATUS "SHLIB_CXX11LDFLAGS variable undefined." )
endif()

set ( PROBABILISTIC_SILENCE_WARNS "$ENV{PROBABILISTIC_SILENCE_WARNS}" )
get_directory_property(hasParent PARENT_DIRECTORY)
if ( hasParent )
    set ( PROBABILISTIC_SILENCE_WARNS "${PROBABILISTIC_SILENCE_WARNS}" PARENT_SCOPE )
endif()
if ( PROBABILISTIC_SILENCE_WARNS )
    message( STATUS "Silencing known warnings: PROBABILISTIC_SILENCE_WARNS = ${PROBABILISTIC_SILENCE_WARNS}" )
else()
    message( STATUS "Not silencing warnings: PROBABILISTIC_SILENCE_WARNS = ${PROBABILISTIC_SILENCE_WARNS}" )
endif()

include( "build_type.cmake" )

# Position independent code needs to be enabled for shared libraries,
# and for static libraries that are compiled into shared libraries,
# which is every library in this project.
set( CMAKE_POSITION_INDEPENDENT_CODE ON )

# add_library's default library is determined by this option. It is also
# queried by e.g. fftw3
set( BUILD_SHARED_LIBS OFF CACHE INTERNAL "" )

# In earlier versions of cmake, archive flags are used that cause fftw3
# to missing some symbols in R docker containers from rocker. Here, we
# manually specify the ar flags used by more recent versions of cmake.
set( CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> <LINK_FLAGS> qc <TARGET> <OBJECTS>" )
set( CMAKE_C_ARCHIVE_APPEND "<CMAKE_AR> <LINK_FLAGS> q <TARGET> <OBJECTS>" )
set( CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> <LINK_FLAGS> qc <TARGET> <OBJECTS>" )
set( CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> <LINK_FLAGS> q <TARGET> <OBJECTS>" )

# Include R LinkingTo libraries.
if ( CLINK_CPPFLAGS )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLINK_CPPFLAGS}" )
endif()

if ( MINGW )
    # The MinGW compiler currently indicates (via
    # macros WINVER and _WIN32_WINNT) to use the
    # windows api of windows server 2003, despite
    # having capability beyond this. Here we choose
    # to enable the windows api up to windows 7.
    # See https://sourceforge.net/p/mingw-w64/bugs/601/
    # and https://docs.microsoft.com/en-us/cpp/porting/modifying-winver-and-win32-winnt?view=vs-2017
    add_definitions( -DWINVER=0x0601 -D_WIN32_WINNT=0x0601 )
endif()

# Only expose annotated code to clients of the dll/so
# unless in debug mode.
if ( NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" )
    if ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
        add_compile_options(
            -fvisibility=hidden
        )
    endif()
endif()

# Use blas implementation if provided.
add_library( R_blas INTERFACE )
if ( BLAS_LIBS )
    target_link_libraries( R_blas INTERFACE "${BLAS_LIBS}" )
endif()

# Use lapack implementation if provided.
add_library( R_lapack INTERFACE )
if ( LAPACK_LIBS )
    target_link_libraries( R_lapack INTERFACE "${LAPACK_LIBS}" )
endif()

add_library( R_blas_lapack INTERFACE )
target_link_libraries( R_blas_lapack
    INTERFACE R_blas
    INTERFACE R_lapack
)

add_library( Rlib INTERFACE )

# We have provided our own versions of some R headers that #include_next the real versions
# and then undefine macros that expand erroneously in other headers. To have these headers
# take priority over the ones provided by the R distribution, we need to have add the
# target_include_directories of our own versions before those of the R distribution.
# Note that target_link_libraries(Rlib INTERFACE R_support) does not achieve this.
target_include_directories(Rlib INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/R_support/include")

if ( R_INCLUDE_DIR )
    target_include_directories( Rlib INTERFACE "${R_INCLUDE_DIR}" )
else()
    target_include_directories( Rlib INTERFACE "$ENV{R_INCLUDE_DIR}" )
endif()
if ( LIBR )
    target_link_libraries( Rlib INTERFACE "${LIBR}" )
else()
    target_link_libraries( Rlib INTERFACE "$ENV{LIBR}" )
endif()
target_link_libraries( Rlib INTERFACE R_blas_lapack )
target_compile_definitions( Rlib
    INTERFACE R_NO_REMAP    # From R-exts.pdf pg 147: "... [prepend] 'Rf_' to all the
                            # function names used from Rinternals.h and R_ext/Error.h"
                            # This was introduced to prevent clashes with the macros
                            # "BOOST_LOG_TRIVIAL(error)" and "BOOST_LOG_TRIVIAL(warning)"
                            # with the error and warning R macros.
    INTERFACE R_NO_REMAP_RMATH
)

add_library( R_global_libs INTERFACE )
if ( LIBS )
    target_link_libraries( R_global_libs INTERFACE "${LIBS}" )
else()
    target_link_libraries( R_global_libs INTERFACE "$ENV{LIBS}" )
endif()

if (${CMAKE_VERSION} VERSION_LESS "3.16.9")
    set(PROBABILISTIC_PRECOMPILE_HEADERS FALSE)
else()
    set(PROBABILISTIC_PRECOMPILE_HEADERS TRUE)
endif()

include("boost.cmake") # In src/ rather than src/external to avoid paths over 100 chars.
if (MINGW)
    add_subdirectory(libtorch_mingw_deps)
endif()
add_subdirectory(pytorch)
add_subdirectory(libtorch_support)
add_subdirectory(std_specialisations)
add_subdirectory(boost_log_R)
add_subdirectory(R_support)
add_subdirectory(external)
add_subdirectory(log)
add_subdirectory(R_protect_guard)
add_subdirectory(R_rng_guard)
add_subdirectory(data_translation)
add_subdirectory(modelling)

add_library( dll_visibility INTERFACE )
target_include_directories( dll_visibility INTERFACE dll_visibility )

# Create fat .so/.dll with all dependencies.
#add_library( probabilistic MODULE
#    "${CMAKE_CURRENT_SOURCE_DIR}/a_source_file.c"
#)
# All source files defining R accessible functions
# need to be placed together in one cmake library,
# otherwise platform dependent linker flags need
# to be applied to expose the relevent functions
# by the dynamic symbol table.
add_library( probabilistic SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/R_init_libprobabilistic/R_init_libprobabilistic.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_test/src/create_tensor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_data_translation/src/R_list_to_libtorch_dict.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_data_translation/src/libtorch_dict_to_R_list.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/torch_rng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/ARARCHTX.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/Ensemble.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/LogScore.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/CensoredLogScore.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/TickScore.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/serialise.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/parameters.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/average_score.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/draw_observations.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/parameterisation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/forward.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/fit.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/buffers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/sampling_distribution_draws.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/performance_divergence_draws.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/TruncatedKernelCLT.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/src/empirical_coverage.cpp"
)
target_link_libraries( probabilistic
    PRIVATE Rlib
    PRIVATE TorchWrapper
    PRIVATE boost
    PRIVATE boost_log_R
    PRIVATE R_support
    PRIVATE dll_visibility
    PRIVATE log
    PRIVATE R_protect_guard
    PRIVATE R_rng_guard
    PRIVATE data_translation
    PRIVATE modelling
)
target_include_directories( probabilistic
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/libtorch_test/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/R_modelling/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/R_data_translation/include"
)
target_compile_definitions( probabilistic
    PRIVATE R_USE_C99_IN_CXX
)
# Strip unnecessary symbols from dll/so.
if ( NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" )
    if ( STRIP_LIBS )
        separate_arguments( STRIP_LIBS_CMD UNIX_COMMAND "${STRIP_LIBS} libprobabilistic${SHLIB_EXT}" )
        add_custom_command( TARGET probabilistic POST_BUILD
            COMMAND ${STRIP_LIBS_CMD}
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
    elseif ( STRIP_SHARED_LIB )
        separate_arguments( STRIP_LIBS_CMD UNIX_COMMAND "${STRIP_SHARED_LIB} libprobabilistic${SHLIB_EXT}" )
        add_custom_command( TARGET probabilistic POST_BUILD
            COMMAND ${STRIP_LIBS_CMD}
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
    else()
        message( WARNING "STRIP_LIBS and STRIP_SHARED_LIB variables undefined, symbols will remain unstripped." )
    endif()
endif()
if ( PROBABILISTIC_SILENCE_WARNS )
    target_compile_options( probabilistic
        PRIVATE -Wno-deprecated-declarations    # auto_ptr
        PRIVATE -Wno-sign-compare
    )
endif()
set_target_properties(probabilistic PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

include( "print_diagnostics.cmake" )

