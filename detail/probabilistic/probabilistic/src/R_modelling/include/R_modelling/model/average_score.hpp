#ifndef PROBABILISTIC_R_MODELLING_AVERAGE_SCORE_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_AVERAGE_SCORE_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_average_score(
        SEXP models_R,
        SEXP observations_R,
        SEXP scoring_rule_R
    );

    DLL_PUBLIC SEXP R_average_score_out_of_sample(
        SEXP models_R,
        SEXP observations_R,
        SEXP scoring_rule_R,
        SEXP in_sample_times_R
    );
}

#endif

