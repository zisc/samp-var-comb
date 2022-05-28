#ifndef PROBABILISTIC_R_MODELLING_PARAMETERS_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_PARAMETERS_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_parameters(
        SEXP models_R
    );

    DLL_PUBLIC SEXP R_change_parameters(
        SEXP model_to_change_R,
        SEXP model_with_parameters_R
    );
}

#endif

