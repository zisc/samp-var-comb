#ifndef PROBABILISTIC_LIBTORCH_TENSOR_TO_R_LIST_HPP_GUARD
#define PROBABILISTIC_LIBTORCH_TENSOR_TO_R_LIST_HPP_GUARD

#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <torch/torch.h>

SEXP to_R_list(
    const torch::Tensor& tensor,
    R_protect_guard& protect_guard
);

struct PersistentArgs {
    torch::Tensor tensor;
    int64_t index_ndim;
    torch::IntArrayRef dimension_sizes;
    std::vector<torch::indexing::TensorIndex> tensor_index;
    std::vector<int*> R_list_index_cols;
    std::vector<double*> R_list_data_cols;
    int64_t current_row = 0;
};

void populate_R_list_rows(
    PersistentArgs& args,
    int64_t this_dimension = 0
);

#endif

