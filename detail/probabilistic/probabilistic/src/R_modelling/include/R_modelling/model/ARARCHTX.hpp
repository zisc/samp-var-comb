#ifndef PROBABILISTIC_R_MODELLING_ARARCH_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_ARARCH_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_ManufactureARARCHTX(
        SEXP parameters_R,
        SEXP buffers_R,
        SEXP observations_dict_R,
        SEXP pre_optimise_R
    );
}

#endif

