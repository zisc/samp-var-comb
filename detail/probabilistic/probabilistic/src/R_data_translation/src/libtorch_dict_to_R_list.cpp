#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <torch/torch.h>
#include <data_translation/libtorch_dict_to_R_list.hpp>
#include <R_data_translation/libtorch_dict_to_R_list.hpp>

SEXP R_libtorch_dict_to_list(SEXP libtorch_dict_R) { return R_handle_exception([&]() {
    auto libtorch_dict = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(libtorch_dict_R);
    R_protect_guard protect_guard;
    return to_R_list(*libtorch_dict, protect_guard);
});}

