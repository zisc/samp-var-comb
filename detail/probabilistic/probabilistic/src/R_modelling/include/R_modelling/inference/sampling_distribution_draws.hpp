#ifndef PROBABILISTIC_R_MODELLING_SAMPLING_DISTRIBUTION_DRAW_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_SAMPLING_DISTRIBUTION_DRAW_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_sampling_distribution_draws(
        SEXP sampling_distribution_R,
        SEXP num_draws_R
    );
}

#endif

