#ifndef PROBABILISTIC_R_MODELLING_DRAW_OBSERVATIONS_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_DRAW_OBSERVATIONS_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_draw_observations(
        SEXP model_R,
        SEXP sample_size_R,
        SEXP burn_in_size_R,
        SEXP first_draw_R
    );
}

#endif

