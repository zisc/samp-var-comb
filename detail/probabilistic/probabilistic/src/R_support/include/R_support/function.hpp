#ifndef PROBABILISTIC_R_SUPPORT_FUNCTION_HPP_GUARD
#define PROBABILISTIC_R_SUPPORT_FUNCTION_HPP_GUARD

#include <vector>
#include <Rinternals.h>
#include <R_protect_guard.hpp>

SEXP call_R_function(const char *fn_string, std::vector<SEXP>& args, R_protect_guard& protect_guard);

#endif

