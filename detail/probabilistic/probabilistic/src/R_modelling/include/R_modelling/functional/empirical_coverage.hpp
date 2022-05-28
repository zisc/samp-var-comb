#ifndef PROBABILISTIC_R_MODELLING_FUNCTIONAL_EMPIRICAL_COVERAGE_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_FUNCTIONAL_EMPIRICAL_COVERAGE_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {

    DLL_PUBLIC SEXP R_empirical_coverage(
        SEXP models_R,
        SEXP data_dict_R,
        SEXP open_lower_probability_R,
        SEXP closed_upper_probability_R,
        SEXP complement_R,
        SEXP in_sample_times_R
    );

    DLL_PUBLIC SEXP R_empirical_coverage_expanding_window_obs(
        SEXP model_R,
        SEXP data_dict_R,
        SEXP open_lower_probability_R,
        SEXP closed_upper_probability_R,
        SEXP complement_R,
        SEXP min_in_sample_times_R
    );

    DLL_PUBLIC SEXP R_empirical_coverage_expanding_window_noobs(
        SEXP model_R,
        SEXP open_lower_probability_R,
        SEXP closed_upper_probability_R,
        SEXP complement_R,
        SEXP min_in_sample_times_R
    );

} 

#endif

