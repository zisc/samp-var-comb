#include <stdexcept>
#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/inference/SamplingDistribution.hpp>
#include <R_modelling/inference/performance_divergence_draws.hpp>

#include <log/trivial.hpp>

SEXP R_performance_divergence_draws(
    SEXP sampling_distribution_R,
    SEXP scoring_rule_R,
    SEXP num_draws_R
) { return R_handle_exception([&]() {
    R_protect_guard protect_guard;
    auto sampling_distribution = EXTPTRSXP_to_shared_ptr<SamplingDistribution>(sampling_distribution_R);
    bool scoring_rule_R_null = Rf_isNull(scoring_rule_R);
    std::shared_ptr<ScoringRule> scoring_rule;
    if (!scoring_rule_R_null) { scoring_rule = EXTPTRSXP_to_shared_ptr<ScoringRule>(scoring_rule_R); }
    auto num_draws = INTEGER(num_draws_R)[0];
    SEXP draws_out = protect_guard.protect(Rf_allocVector(REALSXP, num_draws));
    auto draws_out_a = REAL(draws_out);
    if (!sampling_distribution) {
        throw std::logic_error("!sampling_distribution");
    }
    if (!scoring_rule_R_null && !scoring_rule) {
        throw std::logic_error("!scoring_rule_R_null && !scoring_rule");
    }
    PROBABILISTIC_LOG_TRIVIAL_INFO << "Begin drawing predictive accuracy from a sampling distribution estimate for a \""
                                   << sampling_distribution->get_fit_ref().name() << "\" model.";
    auto performance_divergence_distribution = [&]() {
        if (scoring_rule_R_null) {
            return sampling_distribution->get_performance_divergence_distribution();
        }
        return sampling_distribution->get_performance_divergence_distribution(*scoring_rule);
    }();
    if (!performance_divergence_distribution) {
        throw std::logic_error("!performance_divergence_distribution");
    }
    auto draws_in = performance_divergence_distribution->generate(num_draws, 0, 0.0).front().value();
    auto draws_in_a = draws_in.accessor<double, 1>();
    for (decltype(num_draws) i = 0; i != num_draws; ++i) {
        draws_out_a[i] = draws_in_a[i];
    }
    PROBABILISTIC_LOG_TRIVIAL_INFO << "End drawing.";
    return draws_out;
});}

