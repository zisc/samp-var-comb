#ifndef BOOST_LOG_R_LOG_EXCEPTION_HPP_GUARD
#define BOOST_LOG_R_LOG_EXCEPTION_HPP_GUARD

#include <stdexcept>
#include <R.h>
#include <Rinternals.h>

void R_boost_log_exception(void);
void R_boost_log_exception(std::exception&);

#endif

