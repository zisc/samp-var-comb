old_env()
set(GENERATING_PYTORCH_BUILD FALSE)
set(GENERATING_PYTORCH_BUILD FALSE PARENT_SCOPE)

if (MINGW)
    # Override compile options defined at libtorch/share/cmake/Caffe2/Caffe2Targets.cmake
    # line 64 that assume we are using MSVC.
    set_target_properties(torch_cpu PROPERTIES
        INTERFACE_COMPILE_OPTIONS "\$<\$<COMPILE_LANGUAGE:CXX>:-std=c++14>"
    )
        
    # Remove compiler flags with typos, and reinstate the corrected versions in the
    # TorchWrapperImpl target below.
    set_property(TARGET torch PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()

set(
    ATen_CPU_INCLUDE
    "${CMAKE_CURRENT_BINARY_DIR}/caffe2/aten/src/TH"
    "${CMAKE_CURRENT_SOURCE_DIR}/aten/src/TH"
    "${CMAKE_CURRENT_SOURCE_DIR}/aten/src"
    "${CMAKE_CURRENT_BINARY_DIR}/aten/src"
    "${CMAKE_CURRENT_BINARY_DIR}/caffe2/aten/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/catch/single_include"
    "${CMAKE_CURRENT_BINARY_DIR}/caffe2/aten/src/ATen"
)

add_library(TorchWrapperImpl INTERFACE)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_link_libraries(TorchWrapperImpl INTERFACE -Wl,-force_load torch caffe2::mkl)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(TorchWrapperImpl INTERFACE -Wl,--whole-archive -Wl,--no-as-needed torch caffe2::mkl)
else()
    message(FATAL_ERROR "Unrecognised Compiler.")
endif()
target_include_directories(TorchWrapperImpl INTERFACE
    "${TORCH_SRC_DIR}/csrc"
    "${TORCH_SRC_DIR}/csrc/api"
    "${TORCH_SRC_DIR}/csrc/api/include"
    ${ATen_CPU_INCLUDE}
)
if (MINGW)
    target_link_libraries(TorchWrapperImpl
        INTERFACE libtorch_mingw_deps
    )
    target_compile_options(TorchWrapperImpl INTERFACE -include assert_fail.h)
    
    # Reinstate corrected torch compiler flags here.
    target_compile_definitions(TorchWrapperImpl INTERFACE _GLIBCXX_USE_CXX11_ABI=1)
endif()

# message(FATAL_ERROR "ATen_CPU_INCLUDE = ${ATen_CPU_INCLUDE}")

add_library(TorchWrapper INTERFACE)
target_link_libraries(TorchWrapper
    INTERFACE TorchWrapperImpl
    INTERFACE libtorch_support
)

