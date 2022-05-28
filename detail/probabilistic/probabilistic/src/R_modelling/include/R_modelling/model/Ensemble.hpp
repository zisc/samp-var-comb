#ifndef PROBABILISTIC_R_MODELLING_ENSEMBLE_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_ENSEMBLE_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_ManufactureEnsemble(
        SEXP components_R,
        SEXP parameters_R,
        SEXP buffers_R
    );

    DLL_PUBLIC SEXP R_change_components(
        SEXP ensemble_R,
        SEXP new_components_R
    );
}

#endif

