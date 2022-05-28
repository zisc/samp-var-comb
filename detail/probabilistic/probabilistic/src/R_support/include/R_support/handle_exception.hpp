#ifndef PROBABILISTIC_R_SUPPORT_HANDLE_EXCEPTION_HPP_GUARD
#define PROBABILISTIC_R_SUPPORT_HANDLE_EXCEPTION_HPP_GUARD

#include <exception>
#include <Rinternals.h>
#include <boost_log_R/log_exception.hpp>

template<class T>
SEXP R_handle_exception(T&& function) {
    try {
        return function();
    } catch (std::exception& e) {
        R_boost_log_exception(e);
        return R_NilValue;
    } catch (...) {
        R_boost_log_exception();
        return R_NilValue;
    }
}

#endif

