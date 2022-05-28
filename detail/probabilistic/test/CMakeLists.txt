cmake_minimum_required( VERSION 3.0 )

add_subdirectory(external)

# All source files need to be compiled together in the one
# cmake exectuable. If this isn't the case, platform dependent
# linker flags must be passed to ensure relevent symbols
# are exposed. This also isn't currently easy to do with cmake.
add_executable( tests
    "dummy/src/dummy.cpp"
    "seed_torch_rng.cpp"
    "libtorch/src/create_tensor.cpp"
    "libtorch_support/src/reparameterisations_tests.cpp"
    "libtorch_support/src/Parameterisation_tests.cpp"
    "libtorch_support/src/derivatives_tests.cpp"
    "libtorch_support/src/erfcxs_tests.cpp"
    "libtorch_support/src/logsubexp_tests.cpp"
    "libtorch_support/src/standard_normal_log_cdf_tests.cpp"
    "modelling/distribution/src/Normal_tests.cpp"
    "modelling/distribution/src/Mixture_tests.cpp"
    "modelling/distribution/src/interval_tests.cpp"
    "modelling/model/src/ProbabilisticModule_tests.cpp"
    "test_main.cpp"
)
target_link_libraries( tests
    PUBLIC Rlib
    PUBLIC R_global_libs
    PUBLIC test_boost_test
    PUBLIC libtorch_support
    PUBLIC R_protect_guard
    PUBLIC modelling
)
target_include_directories( tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
if (NOT R_HOME)
    set(R_HOME "$ENV{R_HOME}")
endif()
message("R_HOME=${R_HOME}")
target_compile_definitions( tests
    PRIVATE "PROBABILISTIC_R_HOME=${R_HOME}"
)
set_target_properties( tests PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

