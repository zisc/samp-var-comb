#ifndef PROBABILISTIC_LIBTORCH_SUPPORT_INDEXING_HPP_GUARD
#define PROBABILISTIC_LIBTORCH_SUPPORT_INDEXING_HPP_GUARD

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>

template<class T>
torch::OrderedDict<T, torch::IntArrayRef> sizes(
    const torch::OrderedDict<T, torch::Tensor>& x
) {
    torch::OrderedDict<T, torch::IntArrayRef> out; out.reserve(x.size());
    for (const auto& item : x) {
        out.insert(item.key(), item.value().sizes());
    }
    return out;
}

template<class T1, class T2>
auto sizes(
    const torch::OrderedDict<T1, T2>& x
) {
    torch::OrderedDict<T1, decltype(sizes(x.front().value()))> out; out.reserve(x.size());
    for (const auto& item : x) {
        out.insert(item.key(), sizes(item.value()));
    }
    return out;
}

template<class K, class V>
std::vector<K> get_common_keys(const std::vector<torch::OrderedDict<K, V>>& dicts) {
    std::vector<K> common_keys;

    if (dicts.size() > 0) {
        common_keys = dicts[0].keys();
        std::sort(common_keys.begin(), common_keys.end());
    }

    if (dicts.size() > 1) {
        common_keys.reserve(dicts.size());
        decltype(common_keys) common_keys_intermediary; common_keys_intermediary.reserve(common_keys.size());
        for (auto iter = dicts.cbegin()+1; iter != dicts.cend(); ++iter) {
            auto keys_i = iter->keys();
            std::sort(keys_i.begin(), keys_i.end());
            std::set_intersection(
                common_keys.cbegin(), common_keys.cend(),
                keys_i.cbegin(), keys_i.cend(),
                std::back_inserter(common_keys_intermediary)
            );
            std::swap(common_keys, common_keys_intermediary);
            common_keys_intermediary.clear();
        }
    }

    return common_keys;
}

// This function applies the tensor indexes in idx to x one at a time so that they define slices
// across x rather than masking, see https://github.com/pytorch/pytorch/issues/1080. We first
// index x changing all but the first tensor index in idx to Slice(). Then change all non-ellipsis
// elements in idx to Slice(). Beginning from the second tensor index in the original idx, change
// that back into the original tensor, apply the resulting index to x, and then change back to
// Slice().
torch::Tensor slicing_index(torch::Tensor x, const std::vector<torch::indexing::TensorIndex>& idx);

template<class T>
torch::OrderedDict<T, torch::Tensor> slicing_index(
    const torch::OrderedDict<T, torch::Tensor>& x,
    const std::vector<torch::indexing::TensorIndex>& idx
) {
    torch::OrderedDict<T, torch::Tensor> out; out.reserve(x.size());
    for (const auto& item : x) {
        out.insert(item.key(), slicing_index(item.value(), idx));
    }
    return out;
}

template<class T>
std::vector<torch::OrderedDict<T, torch::Tensor>> slicing_index(
    const torch::OrderedDict<T, torch::Tensor>& x,
    const std::vector<std::vector<torch::indexing::TensorIndex>>& idx
) {
    std::vector<torch::OrderedDict<T, torch::Tensor>> out; out.reserve(idx.size());
    for (const auto& idxi : idx) {
        out.emplace_back(slicing_index(x, idxi));
    }
    return out;
}

template<class T>
void slicing_index_put_(torch::Tensor& x, const std::vector<torch::indexing::TensorIndex>& idx, const T& val) {
    auto numel = x.numel();
    auto one_dim_idx = slicing_index(torch::linspace(0, numel, numel, torch::kLong).reshape_as(x), idx).reshape({-1});
    auto bool_idx = torch::full({numel}, false, torch::kBool);
    bool_idx.index_put_({one_dim_idx}, true);
    bool_idx = bool_idx.reshape_as(x);
    x.index_put_({bool_idx}, val);
}

bool indices_are_equal(
    const torch::indexing::Slice& lhs,
    const torch::indexing::Slice& rhs
);

bool indices_are_equal(
    const torch::indexing::Slice& lhs,
    const torch::Tensor& rhs
);

bool indices_are_equal(
    const torch::Tensor& lhs,
    const torch::indexing::Slice& rhs
);

bool indices_are_equal(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs
);

bool indices_are_equal(
    const torch::indexing::TensorIndex& lhs,
    const torch::indexing::TensorIndex& rhs
);

bool indices_are_equal(
    const std::vector<torch::indexing::TensorIndex>& lhs,
    const std::vector<torch::indexing::TensorIndex>& rhs
);

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs
);

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::Tensor& lhs,
    const torch::indexing::Slice& rhs
);

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::indexing::Slice& lhs,
    const torch::Tensor& rhs
);

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::indexing::Slice& lhs,
    const torch::indexing::Slice& rhs
);

c10::optional<torch::indexing::TensorIndex> intersect(
    const torch::indexing::TensorIndex& lhs,
    const torch::indexing::TensorIndex& rhs
);

c10::optional<torch::indexing::TensorIndex> complement(
    const torch::Tensor& x
);

c10::optional<torch::indexing::TensorIndex> complement(
    const torch::indexing::Slice& x,
    int64_t size
);

c10::optional<torch::indexing::TensorIndex> complement(
    const torch::indexing::TensorIndex& x,
    int64_t size
);

std::vector<std::vector<torch::indexing::TensorIndex>> intersect(
    const std::vector<torch::indexing::TensorIndex>& lhs,
    const std::vector<torch::indexing::TensorIndex>& rhs
);

// This returns the complement of x partitioned into non-overlapping blocks.
// This is done in the standard way in e.g. measure theory:
// A||B = A || (B&&A').
std::vector<std::vector<torch::indexing::TensorIndex>> complement(
    const std::vector<torch::indexing::TensorIndex>& x,
    torch::IntArrayRef sizes
);

std::vector<std::vector<torch::indexing::TensorIndex>> remove_duplicates(
    const std::vector<std::vector<torch::indexing::TensorIndex>>& x
);

std::vector<std::vector<torch::indexing::TensorIndex>> pairwise_intersections(
    const std::vector<std::vector<torch::indexing::TensorIndex>>& lhs,
    const std::vector<std::vector<torch::indexing::TensorIndex>>& rhs
);

// Partition the space implied by sizes (the second argument) by
// x and its complement. If x is already the entire space, simply
// return x.
std::vector<std::vector<torch::indexing::TensorIndex>> partition(
    const std::vector<torch::indexing::TensorIndex>& x,
    torch::IntArrayRef sizes
);

// Partition by group membership, such that all elements in the same group in
// the output appear in the same collection of groups in the input x.
std::vector<std::vector<torch::indexing::TensorIndex>> partition(
    const std::vector<std::vector<torch::indexing::TensorIndex>>& x,
    torch::IntArrayRef sizes
);

template<class T>
std::vector<std::vector<torch::indexing::TensorIndex>> partition(
    const torch::OrderedDict<T, std::vector<torch::indexing::TensorIndex>>& x,
    torch::IntArrayRef sizes
) {
    return partition(x.values(), sizes);
}

template<class T>
std::vector<std::vector<torch::indexing::TensorIndex>> partition(
    torch::OrderedDict<T, std::vector<std::vector<torch::indexing::TensorIndex>>> x,
    torch::IntArrayRef sizes
) {
    std::vector<std::vector<torch::indexing::TensorIndex>> x_values_cat;
    int64_t x_values_cat_size = 0;
    for (const auto& item : x) {
        x_values_cat_size += item.value().size();
    }
    x_values_cat.reserve(x_values_cat_size);

    for (auto& item : x) {
        auto& value = item.value();
        x_values_cat.insert(
            x_values_cat.end(),
            std::make_move_iterator(value.begin()),
            std::make_move_iterator(value.end())
        );
    }

    return partition(x_values_cat, sizes);
}

template<class T1, class T2>
torch::OrderedDict<T1, std::vector<std::vector<torch::indexing::TensorIndex>>> partition(
    const torch::OrderedDict<T1, torch::OrderedDict<T2, std::vector<torch::indexing::TensorIndex>>>& x,
    const torch::OrderedDict<T1, torch::IntArrayRef>& sizes
) {
    torch::OrderedDict<T1, std::vector<std::vector<torch::indexing::TensorIndex>>> out;
    out.reserve(x.size());
    for (const auto& item : x) {
        const auto& key = item.key();
        out.insert(key, partition(item.value(), sizes[key]));
    }
    return out;
}

// This one has an extra nested vector in the type of the argument "x".
template<class T1, class T2>
torch::OrderedDict<T1, std::vector<std::vector<torch::indexing::TensorIndex>>> partition(
    torch::OrderedDict<T1, torch::OrderedDict<T2, std::vector<std::vector<torch::indexing::TensorIndex>>>> x,
    const torch::OrderedDict<T1, torch::IntArrayRef>& sizes
) {
    torch::OrderedDict<T1, std::vector<std::vector<torch::indexing::TensorIndex>>> out;
    out.reserve(x.size());
    for (auto& item : x) {
        const auto& key = item.key();
        out.insert(key, partition(std::move(item.value()), sizes[key]));
    }
    return out;
}

torch::Tensor sample_size(
    const torch::Tensor& observations,
    std::vector<torch::indexing::TensorIndex> index,
    double na_ss = missing::na
);

std::vector<torch::Tensor> sample_size(
    const torch::Tensor& observations,
    const std::vector<std::vector<torch::indexing::TensorIndex>>& indices,
    double na_ss = missing::na
);


template<class T>
torch::OrderedDict<T, std::vector<torch::Tensor>> sample_size(
    const torch::OrderedDict<T, torch::Tensor>& observations,
    const std::vector<std::vector<torch::indexing::TensorIndex>>& indices,
    double na_ss = missing::na
) {
    torch::OrderedDict<T, std::vector<torch::Tensor>> out; out.reserve(observations.size());
    for (const auto& item : observations) {
        out.insert(item.key(), sample_size(item.value(), indices, na_ss));
    }
    return out;
}

template<class T1, class T2, class T3>
auto sample_size(
    const torch::OrderedDict<T1, T2>& observations,
    const torch::OrderedDict<T1, T3>& indices,
    double na_ss = missing::na
) {
    torch::OrderedDict<T1, decltype(sample_size(observations.front().value(), indices.front().value(), na_ss))> out; out.reserve(observations.size());
    for (const auto& item : observations) {
        const auto& item_key = item.key();
        out.insert(item_key, sample_size(item.value(), indices[item_key], na_ss));
    }
    return out;
}

torch::Tensor sample_size_by_element(
    const torch::Tensor& observations,
    const std::vector<std::vector<torch::indexing::TensorIndex>>& indices,
    double na_ss = missing::na
);

template<class T1, class T2, class T3>
auto sample_size_by_element(
    const torch::OrderedDict<T1, T2>& observations,
    const torch::OrderedDict<T1, T3>& indices,
    double na_ss = missing::na
) {
    torch::OrderedDict<T1, decltype(sample_size_by_element(observations.front().value(), indices.front().value(), na_ss))> out; out.reserve(observations.size());
    for (const auto& item : observations) {
        const auto& item_key = item.key();
        out.insert(item_key, sample_size_by_element(item.value(), indices[item_key], na_ss));

        /*
        const auto *indices_item_key = indices.find(item_key);
        if (indices_item_key) {
            out.insert(item_key, sample_size_by_element(item.value(), *indices_item_key, na_ss));
        }
        */
    }
    return out;
}

template<
    class T,
    class F,
    std::enable_if_t<std::is_scalar<T>::value || std::is_convertible<T, torch::Scalar>::value, bool> = true
>
auto elementwise_unary_op(
    T&& x,
    F&& op,
    bool missing_participates = false
) {
    return op(std::forward<T>(x));
}

template<
    class T,
    class F,
    std::enable_if_t<std::is_convertible<T, torch::Tensor>::value, bool> = true
>
auto elementwise_unary_op(
    T&& x,
    F&& op,
    bool missing_participates = false
) {
    if (missing_participates) {
        return op(std::forward<T>(x));
    } else {
        return missing::handle_na(
            [&](const auto& x_nested) {
                return op(x_nested);
            },
            std::forward<T>(x)
        );
    }
}

template<class K, class T, class F>
auto elementwise_unary_op(
    const torch::OrderedDict<K, T>& x,
    F&& op,
    bool missing_participates = false
) {
    torch::OrderedDict<K, decltype(elementwise_unary_op(x.front().value(), op, missing_participates))> out; out.reserve(x.size());
    for (const auto& item : x) {
        out.insert(item.key(), elementwise_unary_op(item.value(), op, missing_participates));
    }
    return out;
}

template<
    class L,
    class R,
    class F,
    std::enable_if_t<
        (std::is_scalar<L>::value || std::is_convertible<L, torch::Scalar>::value) &&
        (std::is_scalar<R>::value || std::is_convertible<R, torch::Scalar>::value),
        bool
    > = true
>
auto elementwise_binary_op(
    L&& lhs,
    R&& rhs,
    F&& op
) {
    return op(std::forward<L>(lhs), std::forward<L>(rhs));
}

template<
    class L,
    class R,
    class F,
    std::enable_if_t<
        (std::is_scalar<L>::value || std::is_convertible<L, torch::Scalar>::value) &&
        std::is_convertible<R, torch::Tensor>::value,
        bool
    > = true
>
auto elementwise_binary_op(
    L&& lhs,
    R&& rhs,
    F&& op
) {
    return missing::handle_na([&](R&& r) { return op(lhs, r); }, std::forward<R>(rhs));
}

template<
    class L,
    class R,
    class F,
    std::enable_if_t<
        std::is_convertible<L, torch::Tensor>::value &&
        (std::is_scalar<R>::value || std::is_convertible<R, torch::Scalar>::value),
        bool
    > = true
>
auto elementwise_binary_op(
    L&& lhs,
    R&& rhs,
    F&& op
) {
    return missing::handle_na([&](L&& l) { return op(l, rhs); }, std::forward<L>(lhs));
}

template<
    class L,
    class R,
    class F,
    std::enable_if_t<std::is_convertible<L, torch::Tensor>::value && std::is_convertible<R, torch::Tensor>::value, bool> = true
>
auto elementwise_binary_op(
    L&& lhs,
    R&& rhs,
    F&& op
) {
    return missing::handle_na(
        [&](L&& l, R&& r) { return op(l, r); },
        std::forward<L>(lhs),
        std::forward<R>(rhs)
    );
}

template<class K, class L, class R, class F>
auto elementwise_binary_op(
    const torch::OrderedDict<K, L>& lhs,
    const torch::OrderedDict<K, R>& rhs,
    F&& op
) {
    auto size = lhs.size();
    if (rhs.size() != size) {
        throw std::logic_error("lhs.size() != rhs.size()");
    }

    torch::OrderedDict<K, decltype(elementwise_binary_op(lhs.front().value(), rhs.front().value(), op))> out; out.reserve(size);
    for (const auto& item : lhs) {
        const auto& key = item.key();
        const auto& lhsi = item.value();
        const auto& rhsi = rhs[key];
        // out.insert(key, op(lhsi, rhsi));
        out.insert(key, elementwise_binary_op(lhsi, rhsi, op));
    }

    return out;
}

template<class K, class V, class F>
auto elementwise_binary_op(
    const torch::Scalar& lhs,
    const torch::OrderedDict<K, V>& rhs,
    F&& op
) {
    auto size = rhs.size();
    torch::OrderedDict<K, decltype(elementwise_binary_op(lhs, rhs.front().value(), op))> out; out.reserve(size);
    for (const auto& item : rhs) {
        const auto& key = item.key();
        const auto& rhsi = item.value();
        // out.insert(key, op(lhs, rhsi));
        out.insert(key, elementwise_binary_op(lhs, rhsi, op));
    }
    return out;
}

template<class K, class V, class F>
auto elementwise_binary_op(
    const torch::OrderedDict<K, V>& lhs,
    const torch::Scalar& rhs,
    F&& op
) {
    auto size = lhs.size();
    torch::OrderedDict<K, decltype(elementwise_binary_op(lhs.front().value(), rhs, op))> out; out.reserve(size);
    for (const auto& item : lhs) {
        const auto& key = item.key();
        const auto& lhsi = item.value();
        // out.insert(key, op(lhsi, rhs));
        out.insert(key, elementwise_binary_op(lhsi, rhs, op));
    }
    return out;
}

template<class K, class L, class R, class F>
auto& elementwise_binary_op_(
    torch::OrderedDict<K, L>& lhs,
    const torch::OrderedDict<K, R>& rhs,
    F&& op
) {
    auto size = lhs.size();
    if (rhs.size() != size) {
        throw std::logic_error("lhs.size() != rhs.size()");
    }

    for (auto& item : lhs) {
        const auto& key = item.key();
        auto& lhsi = item.value();
        const auto& rhsi = rhs[key];
        // op(lhsi, rhsi)
        lhsi = elementwise_binary_op(lhsi, rhsi, op);
    }

    return lhs;
}

template<
    class L,
    class R,
    class F,
    std::enable_if_t<
        (std::is_scalar<L>::value || std::is_convertible<L, torch::Scalar>::value) &&
        (std::is_scalar<R>::value || std::is_convertible<R, torch::Scalar>::value),
        bool
    > = true
>
decltype(auto) elementwise_binary_op_(
    L& lhs,
    R&& rhs,
    F&& op
) {
    return op(lhs, std::forward<R>(rhs));
}

template<
    class L,
    class R,
    class F,
    std::enable_if_t<
        (std::is_scalar<L>::value || std::is_convertible<L, torch::Scalar>::value) &&
        std::is_convertible<R, torch::Tensor>::value,
        bool
    > = true
>
decltype(auto) elementwise_binary_op_(
    L& lhs,
    R&& rhs,
    F&& op
) {
    lhs = missing::handle_na([&](R&& r) { return op(lhs, r); }, rhs);
    return lhs;
}

template<
    class L,
    class R,
    class F,
    std::enable_if_t<
        std::is_convertible<L, torch::Tensor>::value &&
        (std::is_scalar<R>::value || std::is_convertible<R, torch::Scalar>::value),
        bool
    > = true
>
decltype(auto) elementwise_binary_op_(
    L& lhs,
    R&& rhs,
    F&& op
) {
    lhs = missing::handle_na([&](L&& l) { return op(l, rhs); }, lhs);
    return lhs;
}

template<
    class L,
    class R,
    class F,
    std::enable_if_t<std::is_convertible<L, torch::Tensor>::value && std::is_convertible<R, torch::Tensor>::value, bool> = true
>
decltype(auto) elementwise_binary_op_(
    L& lhs,
    R&& rhs,
    F&& op
) {
    lhs = missing::handle_na(
        [&](L&& l, R&& r) { return op(l,r); },
        lhs,
        rhs
    );
    return lhs;
}

/*
template<
    class L,
    class R,
    class F,
    std::enable_if_t<
        (
            std::is_scalar<L>::value ||
            std::is_convertible<L, torch::Tensor>::value
        ) && (
            std::is_scalar<R>::value ||
            std::is_convertible<R, torch::Tensor>::value
        )
    , bool> = true
>
auto& elementwise_binary_op_(
    L& lhs,
    const R& rhs,
    F&& op
) {
    return op(lhs, rhs);
}
*/

namespace torch {

    template<class K, class L, class R>
    auto operator+(
        const torch::OrderedDict<K, L>& lhs,
        const torch::OrderedDict<K, R>& rhs
    ) {
        return elementwise_binary_op(
            lhs,
            rhs,
            /*
            [](const L& l, const R& r) {
                return l+r;
            }
            */
            [](const auto& l, const auto& r) {
                return l+r;
            }
        );
    }

    /*
    template<class T>
    auto& operator+=(
        torch::OrderedDict<T, torch::Tensor>& lhs,
        const torch::OrderedDict<T, torch::Tensor>& rhs
    ) {
        return elementwise_binary_op_(
            lhs,
            rhs,
            [](torch::Tensor& l, const torch::Tensor& r) -> auto& {
                return l.add_(r);
            }
        );
    }
    */

    template<class K, class L, class R>
    auto& operator+=(
        torch::OrderedDict<K, L>& lhs,
        const torch::OrderedDict<K, R>& rhs
    ) {
        for (auto& item : lhs) {
            const auto& key = item.key();
            auto& lhsi = item.value();
            const auto& rhsi = rhs[key];
            // lhsi += rhsi;
            lhsi = elementwise_binary_op(
                lhsi,
                rhsi,
                [](const auto& l, const auto& r) {
                    return l+r;
                }
            );
        }
        return lhs;
    }

    template<class K, class L, class R>
    auto operator*(
        const torch::OrderedDict<K, L>& lhs,
        const torch::OrderedDict<K, R>& rhs
    ) {
        return elementwise_binary_op(
            lhs,
            rhs,
            [](const auto& l, const auto& r) {
                return l*r;
            }
        );
    }

    template<class K, class V>
    auto operator*(
        torch::Scalar k,
        const torch::OrderedDict<K, V>& x
    ) {
        auto mult = [](const auto& l, const auto& r) { return l*r; };
        torch::OrderedDict<K, decltype(elementwise_binary_op(k, x.front().value(), mult))> out; out.reserve(x.size());
        for (const auto& item : x) {
            //out.insert(item.key(), k*item.value());
            out.insert(
                item.key(),
                elementwise_binary_op(
                    k,
                    item.value(),
                    mult
                )
            );
        }
        return out;
    }

    template<class K, class V>
    auto operator*(
        const torch::OrderedDict<K, V>& x,
        torch::Scalar k
    ) {
        return k*x;
    }

    template<class K, class L, class R>
    auto operator/(
        const torch::OrderedDict<K, L>& lhs,
        const torch::OrderedDict<K, R>& rhs
    ) {
        return elementwise_binary_op(
            lhs,
            rhs,
            [](const auto& l, const auto& r) {
                return l/r;
            }
        );
    }

    template<class K, class V>
    auto operator/(
        const torch::OrderedDict<K, V>& lhs,
        torch::Scalar rhs
    ) {
        auto div = [](const auto& l, const auto& r) { return l/r; };
        torch::OrderedDict<K, decltype(elementwise_binary_op(lhs.front().value(), rhs, div))> out; out.reserve(lhs.size());
        for (const auto& item : lhs) {
            //out.insert(item.key(), item.value()/rhs);
            out.insert(
                item.key(),
                elementwise_binary_op(
                    item.value(),
                    rhs,
                    div
                )
            );
        }
        return out;
    }

    template<class K, class V>
    auto operator-(const torch::OrderedDict<K, V>& x) {
        torch::OrderedDict<K, V> negative_x; negative_x.reserve(x.size());
        for (const auto& item : x) {
            // negative_x.insert(item.key(), -item.value());
            negative_x.insert(
                item.key(),
                elementwise_unary_op(
                    item.value(),
                    [](const auto& x) { return -x; }
                )
            );
        }
        return negative_x;
    }

}

template<class T1, class T2>
auto transpose(const torch::OrderedDict<T1, torch::OrderedDict<T2, torch::Tensor>>& x) {
    auto x_size = x.size();
    torch::OrderedDict<T2, torch::OrderedDict<T1, torch::Tensor>> out; out.reserve(x.front().value().size());
    for (const auto& item_i : x) {
        const auto& key_i = item_i.key();
        const auto& xi = item_i.value();
        for (const auto& item_j : xi) {
            const auto& key_j = item_j.key();
            const auto& xij = item_j.value();
            if (!out.contains(key_j)) {
                torch::OrderedDict<T1, torch::Tensor> out_j; out_j.reserve(x_size);
                out.insert(key_j, std::move(out_j));
            }
            out[key_j].insert(key_i, torch::transpose(xij, xij.ndimension()-2, xij.ndimension()-1));
        }
    }
    return out;
}

template<class T>
struct CollapsedTensor {
    torch::Tensor tensor;
    torch::OrderedDict<T, torch::indexing::TensorIndex> indices;
};

template<class T>
CollapsedTensor<T> collapse_vector(const torch::OrderedDict<T, torch::Tensor>& x) {
    torch::OrderedDict<T, torch::indexing::TensorIndex> indices_out; indices_out.reserve(x.size());
    int64_t begin = 0;
    auto end = begin;
    for (const auto& item : x) {
        end = begin + item.value().sizes().back();
        indices_out.insert(item.key(), torch::indexing::Slice(begin, end));
        begin = end;
    }

    auto vector_out = torch::full({end}, std::numeric_limits<double>::quiet_NaN(), torch::kDouble);
    for (const auto& item : x) {
        vector_out.index_put_({indices_out[item.key()]}, item.value());
    }

    return {vector_out, indices_out};
}

template<class T>
CollapsedTensor<T> collapse_matrix(const torch::OrderedDict<T, torch::OrderedDict<T, torch::Tensor>>& x) {
    auto size = x.size();
    for (const auto& item : x) {
        if (item.value().size() != size) {
            throw std::logic_error("item.value().size() != size");
        }
    }

    torch::OrderedDict<T, torch::indexing::TensorIndex> indices_out; indices_out.reserve(size);
    const auto& block_row = x.front().value();
    int64_t begin = 0;
    auto end = begin;
    for (const auto& item : block_row) {
        end = begin + item.value().sizes().back();
        indices_out.insert(item.key(), torch::indexing::Slice(begin, end));
        begin = end;
    }

    auto matrix_out = torch::full({end, end}, std::numeric_limits<double>::quiet_NaN(), torch::kDouble);
    for (const auto& item_i : x) {
        const auto& key_i = item_i.key();
        const auto& xi = item_i.value();
        const auto& idx_i = indices_out[key_i];
        for (const auto& item_j : xi) {
            const auto& key_j = item_j.key();
            const auto& xij = item_j.value();
            const auto& idx_j = indices_out[key_j];
            matrix_out.index_put_({idx_i, idx_j}, xij);
        }
    }

    return {matrix_out, indices_out};
}

#endif

