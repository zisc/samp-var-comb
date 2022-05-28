#include <string>
#include <R.h>
#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <torch/torch.h>
#include <modelling/model/ProbabilisticModule.hpp>
#include <R_modelling/forward.hpp>

SEXP R_forward(
    SEXP model_R,
    SEXP observations_dict_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;
    auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(model_R);
    auto observations = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(observations_dict_R);
    return model->forward(*observations)->to_R_list(protect_guard);
});}

