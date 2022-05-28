set( test_boost_libs_dir "${CMAKE_CURRENT_SOURCE_DIR}/boost/libs" )

add_library( test_boost_include INTERFACE )
target_include_directories( test_boost_include
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/boost"
)

set( test_boost_test_src "${test_boost_libs_dir}/test/src" )
add_library( test_boost_test STATIC
    "${test_boost_test_src}/compiler_log_formatter.cpp"
    "${test_boost_test_src}/debug.cpp"
    "${test_boost_test_src}/decorator.cpp"
    "${test_boost_test_src}/execution_monitor.cpp"
    "${test_boost_test_src}/framework.cpp"
    "${test_boost_test_src}/plain_report_formatter.cpp"
    "${test_boost_test_src}/progress_monitor.cpp"
    "${test_boost_test_src}/results_collector.cpp"
    "${test_boost_test_src}/results_reporter.cpp"
    "${test_boost_test_src}/test_framework_init_observer.cpp"
    "${test_boost_test_src}/test_tools.cpp"
    "${test_boost_test_src}/test_tree.cpp"
    "${test_boost_test_src}/unit_test_log.cpp"
    "${test_boost_test_src}/unit_test_main.cpp"
    "${test_boost_test_src}/unit_test_monitor.cpp"
    "${test_boost_test_src}/unit_test_parameters.cpp"
    "${test_boost_test_src}/junit_log_formatter.cpp"
    "${test_boost_test_src}/xml_log_formatter.cpp"
    "${test_boost_test_src}/xml_report_formatter.cpp"
)
target_link_libraries( test_boost_test
    PUBLIC test_boost_include
    PUBLIC boost_include
    PRIVATE test_boost_timer
)
target_compile_definitions( test_boost_test
    INTERFACE BOOST_TEST_NO_AUTO_LINK=1
    PUBLIC BOOST_TEST_STATIC_LINK=1
)
set_target_properties( test_boost_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)
if ( PROBABILISTIC_SILENCE_WARNS )
    target_compile_options( test_boost_test
        PUBLIC -Wno-deprecated-declarations    # auto_ptr
    )
endif()

set( test_boost_timer_src "${test_boost_libs_dir}/timer/src" )
add_library( test_boost_timer STATIC
    "${test_boost_timer_src}/auto_timers_construction.cpp"
    "${test_boost_timer_src}/cpu_timer.cpp"
)
target_link_libraries( test_boost_timer
    PUBLIC test_boost_include
    PUBLIC boost_include
    PRIVATE boost_chrono
    PRIVATE boost_system
)
target_compile_definitions( test_boost_timer
    PUBLIC BOOST_TIMER_STATIC_LINK=1
)
set_target_properties( test_boost_timer PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)
