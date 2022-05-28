cmake_minimum_required(VERSION 3.0)

set(data_translation_src "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_library(data_translation STATIC
    "${data_translation_src}/libtorch_tensor_to_R_list.cpp"
    "${data_translation_src}/libtorch_dict_to_R_list.cpp"
)
target_include_directories(data_translation
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(data_translation
    PUBLIC libtorch_support
    PUBLIC R_protect_guard
    PUBLIC Rlib
    PRIVATE boost_log_R
)
set_target_properties(data_translation PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

