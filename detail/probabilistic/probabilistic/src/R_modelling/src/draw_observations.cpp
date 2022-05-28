#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <modelling/model/ProbabilisticModule.hpp>
#include <R_modelling/model/draw_observations.hpp>

SEXP R_draw_observations(
    SEXP model_R,
    SEXP sample_size_R,
    SEXP burn_in_size_R,
    SEXP first_draw_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;
    auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(model_R);
    auto sample_size = INTEGER(sample_size_R)[0];
    auto burn_in_size = INTEGER(burn_in_size_R)[0];
    auto first_draw = REAL(first_draw_R)[0];
    return shared_ptr_to_EXTPTRSXP(
        std::make_shared<torch::OrderedDict<std::string, torch::Tensor>>(
            model->draw_observations(sample_size, burn_in_size, first_draw)
        ),
        protect_guard
    );
});}

