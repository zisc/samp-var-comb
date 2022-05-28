cmake_minimum_required(VERSION 3.0)

set(modelling_src "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_library(modelling STATIC
    "${modelling_src}/ShapelyModule.cpp"
    "${modelling_src}/ProbabilisticModule.cpp"
    "${modelling_src}/Distribution.cpp"
    "${modelling_src}/SamplingDistribution.cpp"
    "${modelling_src}/Normal.cpp"
    "${modelling_src}/NormalVector.cpp"
    "${modelling_src}/LogNormal.cpp"
    "${modelling_src}/TranslatedDistribution.cpp"
    "${modelling_src}/Quadratic.cpp"
    "${modelling_src}/Mixture.cpp"
    "${modelling_src}/LogScore.cpp"
    "${modelling_src}/CensoredLogScore.cpp"
    "${modelling_src}/TickScore.cpp"
    "${modelling_src}/ScoringRule.cpp"
    "${modelling_src}/fit.cpp"
    "${modelling_src}/AutoRegressive.cpp"
    "${modelling_src}/ARARCHTX.cpp"
    "${modelling_src}/Ensemble.cpp"
    "${modelling_src}/sample_size.cpp"
    "${modelling_src}/TruncatedKernelCLT.cpp"
    "${modelling_src}/window_average.cpp"
    "${modelling_src}/empirical_coverage.cpp"
)
target_link_libraries(modelling
    PUBLIC std_specialisations
    PUBLIC Rlib
    PUBLIC log
    PUBLIC data_translation
    PUBLIC libtorch_support
    PUBLIC R_rng_guard
    PUBLIC R_protect_guard
    PUBLIC R_support
    PRIVATE boost
)
target_include_directories(modelling
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
if (PROBABILISTIC_PRECOMPILE_HEADERS)
    target_precompile_headers(modelling REUSE_FROM libtorch_support)
endif()
set_target_properties(modelling PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

