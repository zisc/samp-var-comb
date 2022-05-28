#ifndef PROBABILISTIC_R_MODELLING_FORWARD_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_FORWARD_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_forward(
        SEXP model_R,
        SEXP observations_dict_R
    );
}

#endif

