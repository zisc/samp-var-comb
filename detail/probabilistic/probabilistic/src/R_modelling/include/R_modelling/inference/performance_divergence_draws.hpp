#ifndef PROBABILISTIC_R_MODELLING_PERFORMANCE_DIVERGENCE_DRAWS_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_PERFORMANCE_DIVERGENCE_DRAWS_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_performance_divergence_draws(
        SEXP sampling_distribution_R,
        SEXP scoring_rule_R,
        SEXP num_draws_R
    );
}

#endif

