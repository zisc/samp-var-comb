#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <modelling/model/ProbabilisticModule.hpp>
#include <R_modelling/model/average_score.hpp>

/*
SEXP R_average_score(
    SEXP model_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;
    auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(model_R);
    auto observations = model->observations();
    SEXP average_score_R = protect_guard.protect(Rf_allocVector(REALSXP, 1));
    REAL(average_score_R)[0] = model->scoring_rule()->average(
        *(model->forward(observations)),
        observations
    ).item<double>();
    return average_score_R;
});}
*/

/*
SEXP R_average_score(
    SEXP model_R,
    SEXP observations_R,
    SEXP scoring_rule_R
) { return R_handle_exception([&]() {
    R_protect_guard protect_guard;
    auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(model_R);
    const auto& observations = Rf_isNull(observations_R) ? model->observations() : *EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(observations_R);
    auto scoring_rule = Rf_isNull(scoring_rule_R) ? model->scoring_rule() : EXTPTRSXP_to_shared_ptr<ScoringRule>(scoring_rule_R);
    SEXP average_score_R = protect_guard.protect(Rf_allocVector(REALSXP, 1));
    REAL(average_score_R)[0] = scoring_rule->average(
        *(model->forward(observations)),
        observations
    ).item<double>();
    return average_score_R;
});}
*/

SEXP R_average_score(
    SEXP models_R,
    SEXP observations_R,
    SEXP scoring_rule_R
) { return R_handle_exception([&]() {
    R_protect_guard protect_guard;

    std::shared_ptr<torch::OrderedDict<std::string, torch::Tensor>> observations_arg;
    if (!Rf_isNull(observations_R)) { observations_arg = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(observations_R); }

    std::shared_ptr<const ScoringRule> scoring_rule_arg;
    if (!Rf_isNull(scoring_rule_R)) { scoring_rule_arg = EXTPTRSXP_to_shared_ptr<ScoringRule>(scoring_rule_R); }

    int64_t nmodels = Rf_length(models_R);

    SEXP ret_R = protect_guard.protect(Rf_allocVector(REALSXP, nmodels));
    double *ret = REAL(ret_R);
    for (int64_t i = 0; i != nmodels; ++i) {
        auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(VECTOR_ELT(models_R, i));
        const auto& observations = observations_arg ? *observations_arg : model->observations();
        const auto& scoring_rule = scoring_rule_arg ? *scoring_rule_arg : *(model->scoring_rule());
        ret[i] = scoring_rule.average(*(model->forward(observations)), observations).item<double>();
    }

    return ret_R;
});}

SEXP R_average_score_out_of_sample(
    SEXP models_R,
    SEXP observations_R,
    SEXP scoring_rule_R,
    SEXP in_sample_times_R
) { return R_handle_exception([&]() {
    R_protect_guard protect_guard;

    std::shared_ptr<torch::OrderedDict<std::string, torch::Tensor>> observations_arg;
    if (!Rf_isNull(observations_R)) { observations_arg = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(observations_R); }

    std::shared_ptr<const ScoringRule> scoring_rule_arg;
    if (!Rf_isNull(scoring_rule_R)){ scoring_rule_arg = EXTPTRSXP_to_shared_ptr<ScoringRule>(scoring_rule_R); }

    int in_sample_times = INTEGER(in_sample_times_R)[0];

    int64_t nmodels = Rf_length(models_R);

    SEXP ret_R = protect_guard.protect(Rf_allocVector(REALSXP, nmodels));
    double *ret = REAL(ret_R);
    for (int64_t i = 0; i != nmodels; ++i) {
        auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(VECTOR_ELT(models_R, i));
        const auto& observations = observations_arg ? *observations_arg : model->observations();
        const auto& scoring_rule = scoring_rule_arg ? *scoring_rule_arg : *(model->scoring_rule());
        ret[i] = scoring_rule.average_out_of_sample(*(model->forward(observations)), observations, in_sample_times).item<double>();
    }

    return ret_R;
});}

