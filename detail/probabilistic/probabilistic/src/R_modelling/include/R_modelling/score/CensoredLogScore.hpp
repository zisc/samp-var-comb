#ifndef PROBABILISTIC_R_MODELLING_CENSOREDLOGSCORE_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_CENSOREDLOGSCORE_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_ManufactureCensoredLogScore(
        SEXP open_lower_bound,
        SEXP closed_upper_bound,
        SEXP complement
    );

    DLL_PUBLIC SEXP R_ManufactureProbabilityCensoredLogScore(
        SEXP open_lower_probability,
        SEXP closed_upper_probability,
        SEXP complement
    );
}

#endif

