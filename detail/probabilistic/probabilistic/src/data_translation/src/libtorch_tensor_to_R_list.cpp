#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <data_translation/libtorch_tensor_to_R_list.hpp>

void populate_R_list_rows(
    PersistentArgs& args,
    int64_t this_dimension
) {
    if (this_dimension != args.index_ndim) {
        // Loops, this_dimension is an index dimension.
        for (int64_t i = 0; i != args.dimension_sizes.at(this_dimension); ++i) {
            args.tensor_index.at(this_dimension) = i;
            populate_R_list_rows(args, this_dimension + 1);
        }
    } else {
        // Inner loop body, this_dimension is the final dimension along which
        // the data columns in the R list lie, and we may now populate this row
        // of the R_list.

        // Populate columns in R list containing the tensor's indices.
        for (int64_t j = 0; j != args.index_ndim; ++j) {
            args.R_list_index_cols.at(j)[args.current_row] = args.tensor_index.at(j).integer();
        }

        // Populate R_list_data.
        for (int64_t j = 0; j != args.dimension_sizes.at(this_dimension); ++j) {
            args.tensor_index.at(this_dimension) = j;
            args.R_list_data_cols.at(j)[args.current_row] = static_cast<torch::Tensor>(args.tensor.index(args.tensor_index)).item<double>();
        }

        ++args.current_row;
    }
}

SEXP to_R_list(
    const torch::Tensor& tensor,
    R_protect_guard& protect_guard
) {
    PersistentArgs args;
    
    auto expanded_sizes = tensor.sizes().vec();
    expanded_sizes.emplace_back(1);
    args.tensor = missing::replace_na(tensor, NA_REAL).view(expanded_sizes);
    args.index_ndim = args.tensor.ndimension()-1;
    args.dimension_sizes = args.tensor.sizes();
    args.tensor_index.resize(args.tensor.ndimension(), 0);

    int64_t R_list_data_cols = 1;
    int64_t R_list_ncols = args.index_ndim + R_list_data_cols;
    int64_t R_list_nrows = 1;
    for (int64_t i = 0; i != args.index_ndim; ++i) {
        R_list_nrows *= args.tensor.sizes().at(i);
    }

    SEXP ans = protect_guard.protect(Rf_allocVector(VECSXP, R_list_ncols));

    args.R_list_index_cols.reserve(args.index_ndim);
    for (int64_t i = 0; i != args.index_ndim; ++i) {
        SEXP col = Rf_allocVector(INTSXP, R_list_nrows);
        SET_VECTOR_ELT(ans, i, col);
        args.R_list_index_cols.emplace_back(INTEGER(col));
    }

    args.R_list_data_cols.reserve(R_list_data_cols);
    for (int64_t i = args.index_ndim; i != R_list_ncols; ++i) {
        SEXP col = Rf_allocVector(REALSXP, R_list_nrows);
        SET_VECTOR_ELT(ans, i, col);
        args.R_list_data_cols.emplace_back(REAL(col));
    }

    populate_R_list_rows(args);

    return ans;
}

