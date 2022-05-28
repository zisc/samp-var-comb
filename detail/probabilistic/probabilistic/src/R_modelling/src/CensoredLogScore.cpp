#include <stdexcept>
#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <modelling/score/CensoredLogScore.hpp>
#include <R_modelling/score/CensoredLogScore.hpp>

SEXP R_ManufactureCensoredLogScore(
    SEXP open_lower_bound_R,
    SEXP closed_upper_bound_R,
    SEXP complement_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;
    double open_lower_bound = REAL(open_lower_bound_R)[0];
    double closed_upper_bound = REAL(closed_upper_bound_R)[0];
    bool complement = INTEGER(complement_R)[0];
    return shared_ptr_to_EXTPTRSXP(
        ManufactureCensoredLogScore(
            open_lower_bound,
            closed_upper_bound,
            complement
        ),
        protect_guard
    );
});}

SEXP R_ManufactureProbabilityCensoredLogScore(
    SEXP open_lower_probability_R,
    SEXP closed_upper_probability_R,
    SEXP complement_R
) { return R_handle_exception([&]() {
    R_protect_guard protect_guard;
    double open_lower_probability = REAL(open_lower_probability_R)[0];
    double closed_upper_probability = REAL(closed_upper_probability_R)[0];
    bool complement = INTEGER(complement_R)[0];
    return shared_ptr_to_EXTPTRSXP(
        ManufactureProbabilityCensoredLogScore(
            open_lower_probability,
            closed_upper_probability,
            complement
        ),
        protect_guard
    );
});}

