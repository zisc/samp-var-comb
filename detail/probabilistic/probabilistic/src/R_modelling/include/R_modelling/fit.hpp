#ifndef PROBABILISTIC_R_MODELLING_FIT_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_FIT_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_fit(
        SEXP models_R,
        SEXP scoring_rule_R,
        SEXP data_dict_R,
        SEXP learning_rate_R,
        SEXP barrier_begin_R,
        SEXP barrier_end_R,
        SEXP barrier_decay_R,
        SEXP tolerance_grad_R,
        SEXP tolerance_change_R,
        SEXP maximum_optimiser_iterations_R,
        SEXP timeout_in_seconds_R,
        SEXP return_diagnostics_R
    );
}

#endif

