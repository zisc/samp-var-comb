#include <stdexcept>
#include <boost/log/trivial.hpp>
#include <Rinternals.h>
#include <R_ext/Print.h>

void R_boost_log_exception(void) {
    try {
        const char *msg = "C++ unrecognised exception.";
        BOOST_LOG_TRIVIAL(error) << msg;
        Rf_error(msg);
    } catch (std::exception& e_inner) {
        Rf_error("C++ unrecognised exception.\nC++ exception (boost log): %s\n", e_inner.what());
    } catch (...) {
        Rf_error("C++ unrecognised exception.\nC++ unrecognised exception (boost log).\n");
    }
}

void R_boost_log_exception(std::exception& e_outer) {
    try {
        BOOST_LOG_TRIVIAL(error) << "C++ exception: " << e_outer.what();
        Rf_error(e_outer.what());
    } catch (std::exception& e_inner) {
        Rf_error("C++ exception: %s C++ exception (boost log): %s\n", e_outer.what(), e_inner.what());
    } catch (...) {
        Rf_error("C++ exception: %s C++ unrecognised exception (boost log).\n", e_outer.what());
    }
}

