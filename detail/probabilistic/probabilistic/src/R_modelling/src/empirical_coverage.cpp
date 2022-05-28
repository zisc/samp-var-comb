#include <string>
#include <R.h>
#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <torch/torch.h>
#include <modelling/functional/empirical_coverage.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/score/ScoringRule.hpp>
#include <R_modelling/functional/empirical_coverage.hpp>

SEXP R_empirical_coverage(
    SEXP models_R,
    SEXP data_dict_R,
    SEXP open_lower_probability_R,
    SEXP closed_upper_probability_R,
    SEXP complement_R,
    SEXP in_sample_times_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;

    auto data = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(data_dict_R);
    double open_lower_probability = REAL(open_lower_probability_R)[0];
    double closed_upper_probability = REAL(closed_upper_probability_R)[0];
    int complement = LOGICAL(complement_R)[0];
    int in_sample_times = INTEGER(in_sample_times_R)[0];

    int64_t nmodels = Rf_length(models_R);

    SEXP ret_R = protect_guard.protect(Rf_allocVector(REALSXP, nmodels));
    double *ret = REAL(ret_R);
    for (int64_t i = 0; i != nmodels; ++i) {
        auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(VECTOR_ELT(models_R, i));
        ret[i] = empirical_coverage(
            model,
            *data,
            open_lower_probability,
            closed_upper_probability,
            complement,
            in_sample_times
        ).item<double>();
    }

    return ret_R;
});}

SEXP R_empirical_coverage_expanding_window_obs(
    SEXP model_R,
    SEXP data_dict_R,
    SEXP open_lower_probability_R,
    SEXP closed_upper_probability_R,
    SEXP complement_R,
    SEXP min_in_sample_times_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;

    auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(model_R);
    auto data = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(data_dict_R);
    double open_lower_probability = REAL(open_lower_probability_R)[0];
    double closed_upper_probability = REAL(closed_upper_probability_R)[0];
    int complement = LOGICAL(complement_R)[0];
    int min_in_sample_times = INTEGER(min_in_sample_times_R)[0];

    SEXP ret_R = protect_guard.protect(Rf_allocVector(REALSXP, 1));
    REAL(ret_R)[0] = empirical_coverage_expanding_window(
        model,
        *data,
        open_lower_probability,
        closed_upper_probability,
        complement,
        min_in_sample_times
    ).item<double>();

    return ret_R;
});}

SEXP R_empirical_coverage_expanding_window_noobs(
    SEXP model_R,
    SEXP open_lower_probability_R,
    SEXP closed_upper_probability_R,
    SEXP complement_R,
    SEXP min_in_sample_times_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;

    auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(model_R);
    double open_lower_probability = REAL(open_lower_probability_R)[0];
    double closed_upper_probability = REAL(closed_upper_probability_R)[0];
    int complement = LOGICAL(complement_R)[0];
    int min_in_sample_times = INTEGER(min_in_sample_times_R)[0];
    
    SEXP ret_R = protect_guard.protect(Rf_allocVector(REALSXP, 1));
    REAL(ret_R)[0] = empirical_coverage_expanding_window(
        model,
        open_lower_probability,
        closed_upper_probability,
        complement,
        min_in_sample_times
   ).item<double>();

    return ret_R;

});}

