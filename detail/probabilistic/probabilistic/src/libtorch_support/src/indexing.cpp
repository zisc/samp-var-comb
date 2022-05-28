#include <algorithm>
#include <stdexcept>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/indexing.hpp>

torch::Tensor slicing_index(torch::Tensor x, const std::vector<torch::indexing::TensorIndex>& idx) {
    // This function applies the tensor indexes in idx to x one at a time so that they define slices
    // across x rather than masking, see https://github.com/pytorch/pytorch/issues/1080. We first
    // index x changing all but the first tensor index in idx to Slice(). Then change all non-ellipsis
    // elements in idx to Slice(). Beginning from the second tensor index in the original idx, change
    // that back into the original tensor, apply the resulting index to x, and then change back to
    // Slice().

    auto moving_idx = idx;
    std::vector<int64_t> tensor_indexed_dimensions; tensor_indexed_dimensions.reserve(idx.size());
    for (int64_t i = 0; i != idx.size(); ++i) {
        if (idx[i].is_tensor()) {
            tensor_indexed_dimensions.emplace_back(i);
            if (tensor_indexed_dimensions.size() >= 2) {
                // For all tensor indices but the first, change to Slice().
                moving_idx[i] = torch::indexing::Slice();
            }
        }
    }

    // This indexing applies all of the non-tensor-based indices to x, along with
    // the first tensor-based index.
    x = x.index(moving_idx);

    if (tensor_indexed_dimensions.size() >= 2) {
        for (int64_t i = 0; i != idx.size(); ++i) {
            if (!moving_idx[i].is_ellipsis()) {
                // Now that we have applied all non-tensor-based indices to x, make
                // them all Slice()s (except the Ellipsis, which is already effectively
                // multiple Slice()s).
                moving_idx[i] = torch::indexing::Slice();
            }
        }

        // Beginning from the second tensor-based index (the first has
        // already been accounted for)...
        for (int64_t j = 1; j != tensor_indexed_dimensions.size(); ++j) {
            auto i = tensor_indexed_dimensions[j];
            
            // ...reinstate each Slice()d tensor-based index...
            moving_idx[i] = idx[i];

            // ...apply to x...
            x = x.index(moving_idx);

            // ...and re-Slice() the tensor-based index.
            moving_idx[i] = torch::indexing::Slice();
        }
    }

    return x;
}

bool indices_are_equal(
    const torch::indexing::Slice& lhs,
    const torch::indexing::Slice& rhs
) {
    return lhs.start() == rhs.start() && lhs.step() == rhs.step() && lhs.stop() == rhs.stop();
}

bool indices_are_equal(
    const torch::indexing::Slice& lhs,
    const torch::Tensor& rhs
) {
    if (rhs.scalar_type() != torch::kBool) {
        throw std::logic_error("indices_are_equal only accepts bool tensors.");
    }

    auto lhs_subset_rhs = static_cast<torch::Tensor>(rhs.index({lhs}).all()).item<bool>();
    auto rhs_subset_lhs = !static_cast<torch::Tensor>(rhs.clone().index_put_({lhs}, false).any()).item<bool>();

    return lhs_subset_rhs && rhs_subset_lhs;
}

bool indices_are_equal(
    const torch::Tensor& lhs,
    const torch::indexing::Slice& rhs
) {
    return indices_are_equal(rhs, lhs);
}

bool indices_are_equal(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs
) {
    if (lhs.scalar_type() != torch::kBool || rhs.scalar_type() != torch::kBool) {
        throw std::logic_error("indices_are_equal only accepts bool tensors.");
    }

    if (lhs.ndimension() != 1 || rhs.ndimension() != 1) {
        throw std::logic_error("indices_are_equal only accepts one dimensional tensors.");
    }

    return static_cast<torch::Tensor>(lhs.eq(rhs).all()).item<bool>();
}

bool indices_are_equal(
    const torch::indexing::TensorIndex& lhs,
    const torch::indexing::TensorIndex& rhs
) {
    auto invalid_inputs = []() {
        throw std::runtime_error("indices_are_equal function only supports tensor and slice based indices.");
    };

    if (lhs.is_tensor()) {
        if (rhs.is_tensor()) {
            return indices_are_equal(lhs.tensor(), rhs.tensor());
        } else if (rhs.is_slice()) {
            return indices_are_equal(lhs.tensor(), rhs.slice());
        } else {
            invalid_inputs();
            return false;
        }
    } else if (lhs.is_slice()) {
        if (rhs.is_tensor()) {
            return indices_are_equal(lhs.slice(), rhs.tensor());
        } else if (rhs.is_slice()) {
            return indices_are_equal(lhs.slice(), rhs.slice());
        } else {
            invalid_inputs();
            return false;
        }
    } else {
        invalid_inputs();
        return false;
    }
}

bool indices_are_equal(
    const std::vector<torch::indexing::TensorIndex>& lhs,
    const std::vector<torch::indexing::TensorIndex>& rhs
) {
    if (lhs.empty()) {
        return rhs.empty();
    }

    if (rhs.empty()) {
        return false;
    }

    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (int64_t i = 0; i != lhs.size(); ++i) {
        if (!indices_are_equal(lhs[i], rhs[i])) {
            return false;
        }
    }

    return true;
}

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs
) {
    auto lhs_sizes = lhs.sizes();
    auto rhs_sizes = rhs.sizes();
    if (
        lhs.scalar_type() == torch::kBool && lhs_sizes.size() == 1 &&
        rhs.scalar_type() == torch::kBool && rhs_sizes.size() == 1 &&
        lhs_sizes[0] == rhs_sizes[0]
    ) {
        auto out = torch::logical_and(lhs, rhs);
        if (static_cast<torch::Tensor>(out.any()).item<bool>()) {
            return out;
        } else {
            return c10::nullopt;
        }
    } else {
        throw std::runtime_error("Intersect failed on tensor arguments. The arguments must be boolean one-dimensional tensors of identical size.");
    }
}

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::Tensor& lhs,
    const torch::indexing::Slice& rhs
) {
    if (lhs.scalar_type() != torch::kBool || lhs.sizes().size() != 1) {
        throw std::runtime_error("Intersect function only supports boolean one-dimensional tensors.");
    }

    if (rhs.step() != 1) {
        throw std::runtime_error("Intersect function only supports slices with step size of one.");
    }

    auto out = lhs.new_full(lhs.sizes(), false);
    out.index_put_({rhs}, lhs.index({rhs}));
    if (static_cast<torch::Tensor>(out.any()).item<bool>()) {
        return out;
    } else {
        return c10::nullopt;
    }
}

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::indexing::Slice& lhs,
    const torch::Tensor& rhs
) {
    return intersect(rhs, lhs);
}

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::indexing::Slice& lhs,
    const torch::indexing::Slice& rhs
) {
    if (lhs.step() != 1 || rhs.step() != 1) {
        throw std::runtime_error("Intersect function only supports slices with step size of one.");
    }

    auto intersect_start = std::max(lhs.start(), rhs.start());
    auto intersect_stop = std::min(lhs.stop(), rhs.stop());

    if (intersect_stop > intersect_start) {
        return torch::indexing::Slice(intersect_start, intersect_stop);
    } else {
        return c10::nullopt;
    }
}

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::indexing::TensorIndex& lhs,
    const torch::indexing::TensorIndex& rhs
) {
    auto invalid_inputs = []() {
        throw std::runtime_error("Intersect function only supports tensor and slice based indices.");
    };

    if (lhs.is_tensor()) {
        if (rhs.is_tensor()) {
            return intersect(lhs.tensor(), rhs.tensor());
        } else if (rhs.is_slice()) {
            return intersect(lhs.tensor(), rhs.slice());
        } else {
            invalid_inputs();
            return c10::nullopt;
        }
    } else if (lhs.is_slice()) {
        if (rhs.is_tensor()) {
            return intersect(lhs.slice(), rhs.tensor());
        } else if (rhs.is_slice()) {
            return intersect(lhs.slice(), rhs.slice());
        } else {
            invalid_inputs();
            return c10::nullopt;
        }
    } else {
        invalid_inputs();
        return c10::nullopt;
    }
}

c10::optional<torch::indexing::TensorIndex> complement(
    const torch::Tensor& x
) {
    if (x.scalar_type() == torch::kBool && x.sizes().size() == 1) {
        auto out = x.logical_not();
        if (static_cast<torch::Tensor>(out.any()).item<bool>()) {
            return out;
        } else {
            return c10::nullopt;
        }
    } else {
        throw std::runtime_error("complement function only supports boolean tensor indices of one dimensions.");
    }
}

c10::optional<torch::indexing::TensorIndex> complement(
    const torch::indexing::Slice& x,
    int64_t size
) {
    auto out = torch::full({size}, false, torch::kBool);
    out.index_put_({x}, true);
    if (static_cast<torch::Tensor>(out.any()).item<bool>()) {
        return out;
    } else {
        return c10::nullopt;
    }
}

c10::optional<torch::indexing::TensorIndex> complement(
    const torch::indexing::TensorIndex& x,
    int64_t size
) {
    if (x.is_tensor()) {
        return complement(x.tensor());
    } else if (x.is_slice()) {
        return complement(x.slice(), size);
    } else {
        throw std::logic_error("complement only supports tensor and slice based indices.");
    }
}

std::vector<std::vector<torch::indexing::TensorIndex>> intersect(
    const std::vector<torch::indexing::TensorIndex>& lhs,
    const std::vector<torch::indexing::TensorIndex>& rhs
) {
    if (lhs.empty() || rhs.empty()) {
        return {};
    }
    
    if (lhs.size() != rhs.size()) {
        throw std::logic_error("intersect function arguments have different sizes, and those arguments must not feature ellipsis.");
    }

    std::vector<torch::indexing::TensorIndex> out; out.reserve(lhs.size());

    for (int64_t i = 0; i != lhs.size(); ++i) {
        auto isect = intersect(lhs[i], rhs[i]);
        if (isect.has_value()) {
            out.emplace_back(isect.value());
        } else {
            return {};
        }
    }

    return {out};
}

// This returns the complement of x partitioned into non-overlapping blocks.
// This is done in the standard way in e.g. measure theory:
// A||B = A || (B&&A').
std::vector<std::vector<torch::indexing::TensorIndex>> complement(
    const std::vector<torch::indexing::TensorIndex>& x,
    torch::IntArrayRef sizes
) {
    if (!x.empty() && sizes.size() != x.size()) {
        throw std::logic_error("complement function arguments have difference sizes.");
    }

    std::vector<torch::indexing::TensorIndex> out_i(sizes.size(), torch::indexing::Slice());

    if (x.empty()) {
        return {out_i};
    }

    std::vector<std::vector<torch::indexing::TensorIndex>> out; out.reserve(x.size());

    for (int64_t i = 0; i != x.size(); ++i) {
        auto c = complement(x[i], sizes[i]);
        if (c.has_value()) {
            out_i[i] = std::move(c).value();
            out.emplace_back(out_i);
            out_i[i] = x[i];
        }
    }

    return out;
}

std::vector<std::vector<torch::indexing::TensorIndex>> remove_duplicates(
    const std::vector<std::vector<torch::indexing::TensorIndex>>& x
) {
    std::vector<int> duplicate(x.size(),0);
    int64_t nduplicates = 0;
    for (int64_t i = 0; i != x.size(); ++i) {
        for (int64_t j = i+1; j != x.size(); ++j) {
            if (indices_are_equal(x[i], x[j])) {
                duplicate[j] = 1;
                ++nduplicates;
            }
        }
    }
    std::vector<std::vector<torch::indexing::TensorIndex>> out; out.reserve(nduplicates);
    for (int64_t i = 0; i != x.size(); ++i) {
        if (!duplicate[i]) {
            out.emplace_back(x[i]);
        }
    }
    return out;
}

std::vector<std::vector<torch::indexing::TensorIndex>> pairwise_intersections(
    const std::vector<std::vector<torch::indexing::TensorIndex>>& lhs,
    const std::vector<std::vector<torch::indexing::TensorIndex>>& rhs
) {
    std::vector<std::vector<torch::indexing::TensorIndex>> out; out.reserve(lhs.size()*rhs.size());
    for (const auto& l : lhs) {
        for (const auto& r : rhs) {
            auto out_lr_vec = intersect(l,r);
            if (!out_lr_vec.empty()) {
                auto& out_lr = out_lr_vec.at(0);
                bool is_duplicate = false;
                for (const auto& o : out) {
                    if (indices_are_equal(out_lr, o)) {
                        is_duplicate = true;
                        break;
                    }
                }
                if (!is_duplicate) {
                    out.emplace_back(std::move(out_lr));
                }
            }
        }
    }
    return out;
}

// Partition the space implied by sizes (the second argument) by
// x and its complement. If x is already the entire space, simply
// return x.
std::vector<std::vector<torch::indexing::TensorIndex>> partition(
    const std::vector<torch::indexing::TensorIndex>& x,
    torch::IntArrayRef sizes
) {
    auto out = complement(x, sizes);
    out.emplace_back(x);
    return out;
}

// Partition by group membership, such that all elements in the same group in
// the output appear in the same collection of groups in the input x.
std::vector<std::vector<torch::indexing::TensorIndex>> partition(
    const std::vector<std::vector<torch::indexing::TensorIndex>>& x,
    torch::IntArrayRef sizes
) {
    auto out = partition(x.front(), sizes);
    for (int64_t i = 1; i < x.size(); ++i) {
        out = remove_duplicates(pairwise_intersections(out, partition(x[i], sizes)));
    }
    return out;
}

torch::Tensor sample_size(
    const torch::Tensor& observations,
    std::vector<torch::indexing::TensorIndex> index,
    double na_ss
) {
    auto index_set_ndimension = index.size();
    index.emplace_back(torch::indexing::Ellipsis);
    auto observations_indexed = slicing_index(observations, index);
    auto observations_indexed_sizes = observations_indexed.sizes();
    std::vector<int64_t> index_set_dim; index_set_dim.reserve(index_set_ndimension);
    std::vector<int64_t> new_shape; new_shape.reserve(index_set_ndimension + 1);
    for (int64_t i = 0; i != index_set_ndimension; ++i) {
        index_set_dim.emplace_back(i);
        new_shape.emplace_back(observations_indexed_sizes.at(i));
    }
    new_shape.emplace_back(-1);
    return observations_indexed.reshape(new_shape)
                               .ne(na_ss)
                               .sum(index_set_dim, false, torch::kLong)
                               .reshape(observations.sizes().back());
}

std::vector<torch::Tensor> sample_size(
    const torch::Tensor& observations,
    const std::vector<std::vector<torch::indexing::TensorIndex>>& indices,
    double na_ss
) {
    std::vector<torch::Tensor> out; out.reserve(indices.size());
    for (const auto& index : indices) {
        out.emplace_back(sample_size(observations,index, na_ss));
    }
    return out;
}

torch::Tensor sample_size_by_element(
    const torch::Tensor& observations,
    const std::vector<std::vector<torch::indexing::TensorIndex>>& indices,
    double na_ss
) {
    auto indices_size = indices.size();
    if (indices_size == 1) {
        return sample_size(observations, indices.front(), na_ss);
    } else {
        auto coordinate_set_dimensions = observations.sizes().back();
        if (indices_size == coordinate_set_dimensions) {
            auto out = torch::empty({coordinate_set_dimensions}, torch::kLong);
            auto out_a = out.accessor<int64_t, 1>();
            for (int64_t i = 0; i != coordinate_set_dimensions; ++i) {
                out_a[i] = sample_size(observations.index({torch::indexing::Ellipsis, i}).unsqueeze(-1), indices[i], na_ss).item<int64_t>();
            }
            return out;
        } else {
            throw std::logic_error("indices_size != 1 || indices_size != coordinate_set_dimensions");
        }
    }
}

