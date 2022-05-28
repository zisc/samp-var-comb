#include <stdexcept>
#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <modelling/score/TickScore.hpp>
#include <R_modelling/score/TickScore.hpp>

SEXP R_ManufactureTickScore(SEXP probability_R) { return R_handle_exception([&](){
    R_protect_guard protect_guard;
    double probability = REAL(probability_R)[0];
    return shared_ptr_to_EXTPTRSXP(ManufactureTickScore(probability), protect_guard);
});}

