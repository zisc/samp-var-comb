cmake_minimum_required(VERSION 3.0)

add_library(libtorch_support STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Buffers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Parameterisation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reparameterisations.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/time_series.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/derivatives.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/indexing.cpp"
    # "${CMAKE_CURRENT_SOURCE_DIR}/src/missing.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/moments.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/erfcx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/logsubexp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/standard_normal_log_cdf.cpp"
)
target_link_libraries( libtorch_support
    PUBLIC TorchWrapperImpl
    PUBLIC std_specialisations
    PRIVATE log
    PRIVATE Faddeeva
)
target_include_directories(libtorch_support
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
if (PROBABILISTIC_PRECOMPILE_HEADERS)
    target_precompile_headers(libtorch_support PUBLIC [["torch/torch.h"]])
endif()
set_target_properties(libtorch_support PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

