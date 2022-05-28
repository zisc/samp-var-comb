# This is an external dependency placed in src
# rather than src/external to prevent some
# paths going over 100 chars, and giving
# warnings as such from R CMD CHECK.

# Note: compile definitions for compiled boost
# libraries were transcribed from their
# build/Jamefile.v2 file.

set( boost_libs_dir "${CMAKE_CURRENT_SOURCE_DIR}/boost/libs" )

add_library( boost_include INTERFACE )
target_link_libraries( boost_include
    INTERFACE Rlib
)
target_include_directories( boost_include
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/boost"
)
target_compile_definitions( boost_include
    # Issue similar to https://github.com/UCL/STIR/issues/209
    INTERFACE BOOST_MATH_DISABLE_FLOAT128
)

set( boost_atomic_src "${boost_libs_dir}/atomic/src" )
add_library( boost_atomic STATIC
    "${boost_atomic_src}/lockpool.cpp"
)
target_link_libraries( boost_atomic
    PUBLIC boost_include
)
target_compile_definitions( boost_atomic
    PUBLIC BOOST_ATOMIC_STATIC_LINK=1
)

set( boost_chrono_src "${boost_libs_dir}/chrono/src" )
add_library( boost_chrono STATIC
    "${boost_chrono_src}/chrono.cpp"
    "${boost_chrono_src}/process_cpu_clocks.cpp"
    "${boost_chrono_src}/thread_clock.cpp"
)
target_link_libraries( boost_chrono
    PUBLIC boost_include
    PUBLIC boost_system
)
target_compile_definitions( boost_chrono
    PUBLIC BOOST_CHRONO_STATIC_LINK=1
)

set( boost_date_time_src "${boost_libs_dir}/date_time/src/gregorian" )
add_library( boost_date_time STATIC
    "${boost_date_time_src}/greg_month.cpp"
    "${boost_date_time_src}/greg_weekday.cpp"
    "${boost_date_time_src}/date_generators.cpp"
)
target_link_libraries( boost_date_time
    PUBLIC boost_include
)
if ( PROBABILISTIC_SILENCE_WARNS )
    target_compile_options( boost_date_time
        PRIVATE -Wno-deprecated-declarations    # auto_ptr
    )
endif()

set( boost_filesystem_src "${boost_libs_dir}/filesystem/src" )
add_library( boost_filesystem STATIC
    "${boost_filesystem_src}/codecvt_error_category.cpp"
    "${boost_filesystem_src}/operations.cpp"
    "${boost_filesystem_src}/path_traits.cpp"
    "${boost_filesystem_src}/path.cpp"
    "${boost_filesystem_src}/portability.cpp"
    "${boost_filesystem_src}/unique_path.cpp"
    "${boost_filesystem_src}/utf8_codecvt_facet.cpp"
    "${boost_filesystem_src}/windows_file_codecvt.cpp"
)
target_link_libraries( boost_filesystem
    PUBLIC boost_include
    PUBLIC boost_system
)
target_compile_definitions( boost_filesystem
    PUBLIC BOOST_FILESYSTEM_STATIC_LINK=1
)
if ( PROBABILISTIC_SILENCE_WARNS )
    target_compile_options( boost_filesystem
        PRIVATE -Wno-deprecated-declarations
    )
endif()

set( boost_locale_src "${boost_libs_dir}/locale/src" )
add_library( boost_locale STATIC
    "${boost_locale_src}/encoding/codepage.cpp"
    "${boost_locale_src}/shared/date_time.cpp"
    "${boost_locale_src}/shared/format.cpp"
    "${boost_locale_src}/shared/formatting.cpp"
    "${boost_locale_src}/shared/generator.cpp"
    "${boost_locale_src}/shared/ids.cpp"
    "${boost_locale_src}/shared/localization_backend.cpp"
    "${boost_locale_src}/shared/message.cpp"
    "${boost_locale_src}/shared/mo_lambda.cpp"
    "${boost_locale_src}/util/codecvt_converter.cpp"
    "${boost_locale_src}/util/default_locale.cpp"
    "${boost_locale_src}/util/info.cpp"
    "${boost_locale_src}/util/locale_data.cpp"
)
target_link_libraries( boost_locale
    PUBLIC boost_include
    PUBLIC boost_thread
    PUBLIC boost_system
    #PUBLIC icu
)
target_compile_definitions( boost_locale
    PRIVATE BOOST_THREAD_NO_LIB=1
    #PRIVATE BOOST_LOCALE_WITH_ICU=1
)
if ( WIN32 )
    target_compile_definitions( boost_locale
        PRIVATE BOOST_LOCALE_NO_POSIX_BACKEND=1
    )
else()
    target_compile_definitions( boost_locale
        PRIVATE BOOST_LOCALE_NO_WINAPI_BACKEND=1
        PRIVATE BOOST_LOCALE_WITH_ICONV=1
    )
    target_link_libraries( boost_locale
        PUBLIC R_global_libs    # Link iconv.
    )
endif()
if ( PROBABILISTIC_SILENCE_WARNS )
    target_compile_options( boost_locale
        PRIVATE -Wno-deprecated-declarations    # auto_ptr
    )
endif()

set( boost_log_src "${boost_libs_dir}/log/src" )
set( boost_log_common_cpp
    "${boost_log_src}/attribute_name.cpp"
    "${boost_log_src}/attribute_set.cpp"
    "${boost_log_src}/attribute_value_set.cpp"
    "${boost_log_src}/code_conversion.cpp"
    "${boost_log_src}/core.cpp"
    "${boost_log_src}/record_ostream.cpp"
    "${boost_log_src}/severity_level.cpp"
    "${boost_log_src}/global_logger_storage.cpp"
    "${boost_log_src}/named_scope.cpp"
    "${boost_log_src}/process_name.cpp"
    "${boost_log_src}/process_id.cpp"
    "${boost_log_src}/thread_id.cpp"
    "${boost_log_src}/timer.cpp"
    "${boost_log_src}/exceptions.cpp"
    "${boost_log_src}/default_attribute_names.cpp"
    "${boost_log_src}/default_sink.cpp"
    "${boost_log_src}/text_ostream_backend.cpp"
    "${boost_log_src}/text_file_backend.cpp"
    "${boost_log_src}/text_multifile_backend.cpp"
    "${boost_log_src}/thread_specific.cpp"
    "${boost_log_src}/once_block.cpp"
    "${boost_log_src}/timestamp.cpp"
    "${boost_log_src}/threadsafe_queue.cpp"
    "${boost_log_src}/event.cpp"
    "${boost_log_src}/trivial.cpp"
    "${boost_log_src}/spirit_encoding.cpp"
    "${boost_log_src}/format_parser.cpp"
    "${boost_log_src}/date_time_format_parser.cpp"
    "${boost_log_src}/named_scope_format_parser.cpp"
    "${boost_log_src}/unhandled_exception_count.cpp"
    "${boost_log_src}/permissions.cpp"
    "${boost_log_src}/dump.cpp"
)
set( boost_log_windows_src "${boost_log_src}/windows" )
set( boost_log_windows_cpp
    "${boost_log_windows_src}/light_rw_mutex.cpp"
)
set( boost_log_setup_src "${boost_log_src}/setup" )
set( boost_log_setup_cpp
    "${boost_log_setup_src}/parser_utils.cpp"
    "${boost_log_setup_src}/init_from_stream.cpp"
    "${boost_log_setup_src}/init_from_settings.cpp"
    "${boost_log_setup_src}/settings_parser.cpp"
    "${boost_log_setup_src}/filter_parser.cpp"
    "${boost_log_setup_src}/formatter_parser.cpp"
    "${boost_log_setup_src}/default_filter_factory.cpp"
    "${boost_log_setup_src}/matches_relation_factory.cpp"
    "${boost_log_setup_src}/default_formatter_factory.cpp"
)
if ( WIN32 )
    add_library( boost_log STATIC
        ${boost_log_common_cpp}
        ${boost_log_windows_cpp}
    )
else()
    add_library( boost_log STATIC
        ${boost_log_common_cpp}
    )
endif()
target_link_libraries( boost_log
    PUBLIC boost_include
    PRIVATE boost_locale
    PRIVATE boost_regex
)
target_compile_definitions( boost_log
    PRIVATE BOOST_LOG_BUILDING_THE_LIB=1
    PRIVATE BOOST_LOG_USE_BOOST_REGEX
    PRIVATE BOOST_LOG_WITHOUT_IPC
    PRIVATE BOOST_LOG_WITHOUT_SYSLOG
    PRIVATE BOOST_LOG_WITHOUT_EVENT_LOG
    PRIVATE __STDC_CONSTANT_MACROS
    PRIVATE BOOST_SPIRIT_USE_PHOENIX_V3=1
    PRIVATE BOOST_THREAD_DONT_USE_CHRONO=1
)
if ( PROBABILISTIC_SILENCE_WARNS )
    target_compile_options( boost_log
        PRIVATE -Wno-deprecated-declarations    # auto_ptr
        PRIVATE -Wno-strict-aliasing
        PRIVATE -Wno-nonnull
    )
endif()
add_library( boost_log_setup STATIC
    ${boost_log_setup_cpp}
)
target_link_libraries( boost_log_setup
    PUBLIC boost_include
    PUBLIC boost_log
    PRIVATE boost_random
    PRIVATE boost_regex
)
target_include_directories( boost_log_setup
    PRIVATE "${boost_log_src}"
)
if ( PROBABILISTIC_SILENCE_WARNS)
    target_compile_options( boost_log_setup
        PRIVATE -Wno-deprecated-declarations    # auto_ptr
        PRIVATE -Wno-strict-aliasing
        PRIVATE -Wno-nonnull
    )
    if ( CMAKE_CXX_COMPILER_ID MATCHES "GNU" )
        target_compile_options( boost_log_setup
            PRIVATE -Wno-maybe-uninitialized
        )
    endif()
endif()

set( boost_regex_src "${boost_libs_dir}/regex/src" )
add_library( boost_regex STATIC
   "${boost_regex_src}/c_regex_traits.cpp"
   "${boost_regex_src}/cpp_regex_traits.cpp"
   "${boost_regex_src}/cregex.cpp"
   "${boost_regex_src}/fileiter.cpp"
   "${boost_regex_src}/icu.cpp"
   "${boost_regex_src}/instances.cpp"
   "${boost_regex_src}/posix_api.cpp"
   "${boost_regex_src}/regex.cpp"
   "${boost_regex_src}/regex_debug.cpp"
   "${boost_regex_src}/regex_raw_buffer.cpp"
   "${boost_regex_src}/regex_traits_defaults.cpp"
   "${boost_regex_src}/static_mutex.cpp"
   "${boost_regex_src}/w32_regex_traits.cpp"
   "${boost_regex_src}/wc_regex_traits.cpp"
   "${boost_regex_src}/wide_posix_api.cpp"
   "${boost_regex_src}/winstances.cpp"
   "${boost_regex_src}/usinstances.cpp"
)
target_link_libraries( boost_regex
    PUBLIC boost_include
    #PRIVATE icu
)
target_compile_definitions( boost_regex
    #PRIVATE BOOST_HAS_ICU=1
    PRIVATE U_STATIC_IMPLEMENTATION=1
)
if ( PROBABILISTIC_SILENCE_WARNS )
    target_compile_options( boost_regex
        PRIVATE -Wno-deprecated-declarations    # auto_ptr
    )
endif()

set( boost_random_src "${boost_libs_dir}/random/src" )
add_library( boost_random STATIC
    "${boost_random_src}/random_device.cpp"
)
target_link_libraries( boost_random
    PUBLIC boost_include
    PUBLIC boost_system
)

# The static lib boost system is resulting in linking
# errors. Using it as a header only library instead.
#set( boost_system_src "${boost_libs_dir}/system/src" )
#add_library( boost_system STATIC
#    "${boost_system_src}/error_code.cpp"
#)
#target_link_libraries( boost_system
#    PUBLIC boost_include
#)
#target_compile_definitions( boost_system
#    PUBLIC BOOST_SYSTEM_STATIC_LINK=1
#)
#set_target_properties( boost_system PROPERTIES
#    CXX_STANDARD 11
#    CXX_STANDARD_REQUIRED ON
#)

add_library( boost_system INTERFACE )
target_compile_definitions( boost_system
    INTERFACE BOOST_ERROR_CODE_HEADER_ONLY
)
target_link_libraries( boost_system
    INTERFACE boost_include
)

set( boost_thread_src "${boost_libs_dir}/thread/src" )
if ( WIN32 )
    add_library( boost_thread STATIC
        "${boost_thread_src}/future.cpp"
        "${boost_thread_src}/win32/thread_primitives.cpp"
        "${boost_thread_src}/win32/thread.cpp"
        "${boost_thread_src}/win32/tss_dll.cpp"
        "${boost_thread_src}/win32/tss_pe.cpp"
    )
    target_compile_definitions( boost_thread
        PUBLIC BOOST_THREAD_WIN32
    )
    target_link_libraries( boost_thread
        PUBLIC boost_chrono
    )
else()
    add_library( boost_thread STATIC
        "${boost_thread_src}/future.cpp"
        "${boost_thread_src}/pthread/once.cpp"
        "${boost_thread_src}/pthread/thread.cpp"
    )
    target_compile_definitions( boost_thread
        PUBLIC BOOST_THREAD_POSIX
    )
    target_link_libraries( boost_thread
        PUBLIC boost_atomic
    )
endif()
target_link_libraries( boost_thread
    PUBLIC boost_include
    PUBLIC boost_date_time
    PUBLIC boost_system
)
target_compile_definitions( boost_thread
    PUBLIC BOOST_THREAD_BUILD_LIB=1
    PUBLIC BOOST_THREAD_USE_LIB=1
)
if ( PROBABILISTIC_SILENCE_WARNS )
    if ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
        target_compile_options( boost_thread
            PRIVATE -Wno-long-long
            PRIVATE -Wno-nonnull
        )
    endif()
    target_compile_options( boost_thread
        PRIVATE -Wno-deprecated-declarations    # auto_ptr
    )
endif()

add_library( boost INTERFACE )
target_link_libraries( boost
    INTERFACE boost_include
    INTERFACE boost_atomic
    INTERFACE boost_filesystem
    INTERFACE boost_locale
    INTERFACE boost_log
    INTERFACE boost_log_setup
    INTERFACE boost_random
    INTERFACE boost_regex
    INTERFACE boost_system
    INTERFACE boost_thread
)

