#include <vector>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <libtorch_support/time_series.hpp>

torch::Tensor lag(const torch::Tensor& x, int64_t t_dim, int64_t order, double na_l) {
    auto x_sizes = x.sizes();
    auto x_sizes_size = x_sizes.size();
    if (t_dim < 0) { t_dim = x_sizes_size + t_dim; }
    auto x_sizes_t_dim = x_sizes.at(t_dim);

    std::vector<torch::indexing::TensorIndex> pad_idx;
    pad_idx.reserve(x_sizes_size);
    for (int64_t i = 0; i != x_sizes_size; ++i) {
        if (i == t_dim) {
            pad_idx.emplace_back(torch::indexing::Slice(0, order));
        } else {
            pad_idx.emplace_back(torch::indexing::Slice());
        }
    }

    std::vector<torch::indexing::TensorIndex> t_idx;
    t_idx.reserve(x_sizes_size);
    for (int64_t i = 0; i != x_sizes_size; ++i) {
        if (i == t_dim) {
            t_idx.emplace_back(torch::indexing::Slice(order, x_sizes_t_dim));
        } else {
            t_idx.emplace_back(torch::indexing::Slice());
        }
    }

    std::vector<torch::indexing::TensorIndex> tmord_idx;
    tmord_idx.reserve(x_sizes_size);
    for (int64_t i = 0; i != x_sizes_size; ++i) {
        if (i == t_dim) {
            tmord_idx.emplace_back(torch::indexing::Slice(0, x_sizes_t_dim - order));
        } else {
            tmord_idx.emplace_back(torch::indexing::Slice());
        }
    }

    auto ret = x.new_empty(x.sizes());
    ret.index_put_(pad_idx, na_l);
    ret.index_put_(t_idx, x.index(tmord_idx));

    return ret;
}

torch::Tensor diff(const torch::Tensor& x, int64_t t_dim) {
    return missing::handle_na(
        [](const torch::Tensor& a, const torch::Tensor& b) {
            return a - b;
        },
        x,
        lag(x, t_dim)
    );
}

