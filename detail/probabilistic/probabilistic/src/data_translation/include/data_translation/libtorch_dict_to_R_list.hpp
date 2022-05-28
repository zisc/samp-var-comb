#ifndef PROBABILISTIC_LIBTORCH_DICT_TO_R_LIST_HPP_GUARD
#define PROBABILISTIC_LIBTORCH_DICT_TO_R_LIST_HPP_GUARD

#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <torch/torch.h>

SEXP to_R_list(
    const torch::OrderedDict<std::string, torch::Tensor>& dict,
    R_protect_guard& protect_guard
);

#endif

