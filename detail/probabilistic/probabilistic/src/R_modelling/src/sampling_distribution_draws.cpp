#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/inference/SamplingDistribution.hpp>
#include <R_modelling/inference/sampling_distribution_draws.hpp>

#include <log/trivial.hpp>

SEXP R_sampling_distribution_draws(
    SEXP sampling_distribution_R,
    SEXP num_draws_R
) { return R_handle_exception([&]() {
    R_protect_guard protect_guard;
    auto sampling_distribution = EXTPTRSXP_to_shared_ptr<SamplingDistribution>(sampling_distribution_R);
    PROBABILISTIC_LOG_TRIVIAL_INFO << "Begin drawing parameters from a sampling distribution estimate for a \""
                                   << sampling_distribution->get_fit_ref().name() << "\" model.";
    auto num_draws = INTEGER(num_draws_R)[0];
    auto ten_percent = num_draws/10;
    SEXP draws = protect_guard.protect(Rf_allocVector(VECSXP, num_draws));
    for (decltype(num_draws) i = 0; i != num_draws; ++i) {
        if ((i+1)%ten_percent == 0) PROBABILISTIC_LOG_TRIVIAL_INFO << i+1 << " of " << num_draws << ".";
        shared_ptr_to_EXTPTRSXP<ProbabilisticModule, torch::nn::Module>(draws, i, sampling_distribution->draw_stochastic_process());        
    }
    return draws;
});}

