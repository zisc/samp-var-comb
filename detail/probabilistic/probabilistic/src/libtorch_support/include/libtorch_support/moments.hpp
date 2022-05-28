#ifndef PROBABILISTIC_LIBTORCH_SUPPORT_MOMENTS_HPP_GUARD
#define PROBABILISTIC_LIBTORCH_SUPPORT_MOMENTS_HPP_GUARD

#include <numeric>
#include <limits>
#include <sstream>
#include <vector>
#include <libtorch_support/missing.hpp>
#include <libtorch_support/indexing.hpp>
#include <libtorch_support/time_series.hpp>
#include <torch/torch.h>

#include <log/trivial.hpp>

torch::Tensor sample_size_present(const torch::Tensor& x_present);

torch::Tensor sample_size(
    const torch::Tensor& x,
    double na_ss = missing::na
);

template<class T1, class T2>
torch::Tensor sample_size_present(
    const torch::OrderedDict<T1, T2>& x_present
) {
    torch::Tensor ret;
    for (const auto& item : x_present) {
        if (ret.numel()) {
            ret += sample_size_present(item.value());
        } else {
            ret = sample_size_present(item.value());
        }
    }
    return ret;
}

template<class T1, class T2>
torch::Tensor sample_size(
    const torch::OrderedDict<T1, T2>& x,
    double na_ss = missing::na
) {
    return sample_size_present(missing::is_present(x, na_ss));
}

torch::Tensor sum_present(
    const torch::Tensor& x,
    const torch::Tensor& x_present
);

torch::Tensor sum(
    const torch::Tensor& x,
    double na_s = missing::na
);

template<class T1, class T2>
torch::Tensor sum_present(
    const torch::OrderedDict<T1, T2>& x,
    const torch::OrderedDict<T1, T2>& x_present
) {
    torch::Tensor ret;
    for (const auto& item : x) {
        auto name = item.key();
        auto xi = item.value();
        auto xi_present = x_present[name];
        if (ret.numel()) {
            ret += sum_present(xi, xi_present);
        } else {
            ret = sum_present(xi, xi_present);
        }
    }
    return ret;
}

template<class T1, class T2>
torch::Tensor sum(
    const torch::OrderedDict<T1, T2>& x,
    double na_s
) {
    return sum_present(x, missing::is_present(x, na_s));
}

torch::Tensor average_present(
    const torch::Tensor& x,
    const torch::Tensor& x_present
);

torch::Tensor average(
    const torch::Tensor& x,
    double na_av = missing::na
);

template<class T1, class T2>
torch::Tensor average_present(
    const torch::OrderedDict<T1, T2>& x,
    const torch::OrderedDict<T1, T2>& x_present,
    double na_avp = missing::na
) {
    auto x_sample_size = sample_size_present(x_present);
    if (static_cast<torch::Tensor>(x_sample_size.eq(0)).item<bool>()) {
        return torch::full({}, na_avp);
    }
    return sum_present(x, x_present)/x_sample_size;
}

template<class T1, class T2>
torch::Tensor average(
    const torch::OrderedDict<T1, T2>& x,
    double na_av = missing::na
) {
    return average_present(x, missing::is_present(x));
}

torch::Tensor average_once_zeroed(
    const torch::Tensor& x_zeroed,
    const std::vector<int64_t>& dim,
    torch::Tensor x_sample_sizes,
    bool keepdim = false
);

torch::Tensor average(
    const torch::Tensor& x,
    const std::vector<int64_t>& dim,
    bool keepdim = false,
    double na_av = missing::na
);

// For cross_covariance_matrix, assume that the last dimension
// in the index identifies the element of a vector which is
// an observation, and the prior dimensions of the index identify
// the observation.

void get_dimensions_across_observations_impl(
    std::vector<int64_t>& dimensions_across_observations,
    int64_t ndimension
);

void get_dimensions_across_observations(
    std::vector<int64_t>& dimensions_across_observations,
    int64_t ndimension
);

void get_dimensions_across_observations(
    std::vector<int64_t>& dimensions_across_observations,
    const torch::Tensor& x
);

torch::Tensor cross_covariance_matrix_once_zeroed(
    const torch::Tensor& outer_products_zeroed,
    const torch::Tensor& outer_products_sample_sizes,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
);

torch::Tensor cross_covariance_matrix_once_zeroed(
    const torch::Tensor& x_centered_zeroed,
    const torch::Tensor& y_centered_zeroed,
    const torch::Tensor& outer_products_sample_sizes,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
);

torch::Tensor cross_covariance_matrix_once_zeroed(
    const torch::Tensor& x_centered_zeroed,
    const torch::Tensor& y_centered_zeroed,
    const torch::Tensor& outer_product_sample_sizes,
    double na_cov = missing::na,
    std::vector<int64_t>&& dimensions_across_observations = {}
);

torch::Tensor cross_covariance_matrix_once_centered(
    const torch::Tensor& x_centered,
    const torch::Tensor& y_centered,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
);

torch::Tensor cross_covariance_matrix_once_centered(
    const torch::Tensor& x_centered,
    const torch::Tensor& y_centered,
    double na_cov = missing::na,
    std::vector<int64_t>&& dimensions_across_observations = {}
);

torch::Tensor cross_covariance_matrix(
    const torch::Tensor& x,
    const torch::Tensor& x_average,
    const torch::Tensor& y,
    const torch::Tensor& y_average,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
);

torch::Tensor cross_covariance_matrix(
    const torch::Tensor& x,
    const torch::Tensor& x_average,
    const torch::Tensor& y,
    const torch::Tensor& y_average,
    double na_cov = missing::na,
    std::vector<int64_t>&& dimensions_across_observations = {}
);

torch::Tensor cross_covariance_matrix(
    const torch::Tensor& x,
    const torch::Tensor& y,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
);

torch::Tensor cross_covariance_matrix(
    const torch::Tensor& x,
    const torch::Tensor& y,
    double na_cov = missing::na,
    std::vector<int64_t>&& dimensions_across_observations = {}
);

template<class T1, class T2>
torch::OrderedDict<T1, torch::OrderedDict<T2, torch::Tensor>> cross_covariance_matrix(
    const torch::OrderedDict<T1, torch::Tensor>& x,
    const torch::OrderedDict<T2, torch::Tensor>& y,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
) {
    torch::OrderedDict<T1, torch::OrderedDict<T2, torch::Tensor>> out;
    out.reserve(x.size());
    auto y_size = y.size();
    for (const auto& x_item : x) {
        const auto& x_item_value = x_item.value();
        torch::OrderedDict<T2, torch::Tensor> out_nested;
        out_nested.reserve(y_size);
        for (const auto& y_item : y) {
            out_nested.insert(
                y_item.key(),
                cross_covariance_matrix(
                    x_item_value,
                    y_item.value(),
                    na_cov,
                    dimensions_across_observations
                )
            );
        }
        out.insert(x_item.key(), std::move(out_nested));
    }
    return out;
}

template<class T1, class T2>
torch::OrderedDict<T1, torch::OrderedDict<T2, torch::Tensor>> cross_covariance_matrix(
    const torch::OrderedDict<T1, torch::Tensor>& x,
    const torch::OrderedDict<T2, torch::Tensor>& y,
    double na_cov = missing::na,
    std::vector<int64_t>&& dimensions_across_observations = {}
) {
    get_dimensions_across_observations(dimensions_across_observations, x.front().value());
    return cross_covariance_matrix(x, y, na_cov, dimensions_across_observations);
}

// Need to double check that using the max sample size is appropriate.
// In fact, we assume that for each parameterisation the sample size is
// either zero or identical for each other parameter (or each element of
// each other parameter vector). We should test this assumption somewhere,
// and throw if it is not met.
inline double truncated_kernel_lag_multiplier(
    const torch::Tensor& x_sample_size,
    int64_t lag
) {
    if (lag < 0) lag = -lag;
    auto max_sample_size = static_cast<torch::Tensor>(x_sample_size.max().toType(torch::kDouble)).item<double>();
    return (max_sample_size - lag)/max_sample_size;
}

template<class T>
double truncated_kernel_lag_multiplier(
    const torch::OrderedDict<T, torch::Tensor>& x_sample_size,
    int64_t lag
) {
    if (lag < 0) lag = -lag;
    double out = 0.0;
    for (const auto& item : x_sample_size) {
        auto candidate = truncated_kernel_lag_multiplier(item.value(), lag);
        if (candidate > out) out = candidate;
    }
    return out;
}

template<class T>
torch::OrderedDict<T, torch::OrderedDict<T, torch::Tensor>> truncated_kernel_asymptotic_covariance_matrix(
    const torch::OrderedDict<T, torch::Tensor>& x,
    int64_t t_dim = -1,
    double na_cov = missing::na,
    bool dependent = true
) {
    // Assume each observation is identified by the index in all
    // but the final dimension.
    torch::OrderedDict<T, std::vector<torch::indexing::TensorIndex>> index_set; index_set.reserve(x.size());
    for (const auto& item : x) {
        index_set.insert(item.key(), std::vector<torch::indexing::TensorIndex>(item.value().ndimension()-1, torch::indexing::Slice()));
    }

    auto x_sample_size = sample_size(x, index_set, na_cov);

    auto min_x_sample_size = std::numeric_limits<int64_t>::max();
    for (const auto& item : x) {
        auto min_candidate = static_cast<torch::Tensor>(item.value().min()).item<int64_t>();
        if (min_candidate < min_x_sample_size) {
            min_x_sample_size = min_candidate;
        }
    }

    auto k = std::llround(std::sqrt(static_cast<double>(min_x_sample_size)));

    auto out = cross_covariance_matrix(x, x, na_cov);
    if (dependent) for (int64_t j = 1; j < k; ++j) {
        auto lag_multiplier = truncated_kernel_lag_multiplier(x_sample_size, j);
        auto cov_x_x_lagged_t = cross_covariance_matrix(x, lag(x, t_dim, j, na_cov));
        out += lag_multiplier*(cov_x_x_lagged_t + transpose(cov_x_x_lagged_t));
    }

    return out;
}

template<class T>
torch::OrderedDict<T, torch::OrderedDict<T, torch::Tensor>> truncated_kernel_asymptotic_covariance_matrix(
    const torch::OrderedDict<T, torch::Tensor>& x,
    std::vector<torch::indexing::TensorIndex> index,
    int64_t t_dim = -1,
    double na_cov = missing::na
) {
    index.emplace_back(torch::indexing::Ellipsis);
    return truncated_kernel_asymptotic_covariance_matrix(
        slicing_index(x, index),
        t_dim,
        na_cov
    );
}

template<class T>
std::vector<torch::OrderedDict<T, torch::OrderedDict<T, torch::Tensor>>> trunctated_kernel_asymptotic_covariance_matrix(
    const torch::OrderedDict<T, torch::Tensor>& x,
    std::vector<std::vector<torch::indexing::TensorIndex>> indices,
    int64_t t_dim = -1,
    double na_cov = missing::na
) {
    auto indices_size = indices.size();
    std::vector<torch::OrderedDict<T, torch::OrderedDict<T, torch::Tensor>>> out; out.reserve(indices_size);
    for (int64_t i = 0; i != indices_size; ++i) {
        out.emplace_back(truncated_kernel_asymptotic_covariance_matrix(
            x,
            std::move(indices[i]),
            t_dim,
            na_cov
        ));
    }
    return out;
}

int64_t reduce_sample_size_by_parameterisation(
    const torch::Tensor& sample_size_to_reduce
);

std::vector<int64_t> reduce_sample_size_by_parameterisation(
    const std::vector<torch::Tensor>& sample_size_to_reduce
);

template<class T>
std::vector<int64_t> reduce_sample_size_by_parameterisation_from_estimating_equations(
    const torch::OrderedDict<T, std::vector<torch::Tensor>>& sample_size_to_reduce
) {
    std::vector<int64_t> ret;
    for (const auto& item : sample_size_to_reduce) {
        if (ret.empty()) {
            ret = reduce_sample_size_by_parameterisation(item.value());
        } else {
            auto ret_candidates = reduce_sample_size_by_parameterisation(item.value());
            auto size = ret.size();
            if (ret_candidates.size() != size) {
                throw std::logic_error("ret_candidates.size() != ret.size()");
            }
            for (decltype(size) i = 0; i != size; ++i) {
                auto candidate = ret_candidates[i];
                auto incumbent = ret[i];
                if (candidate) {
                    if (incumbent) {
                        if (candidate != incumbent) {
                            PROBABILISTIC_LOG_TRIVIAL_DEBUG << item.key() << " = parameter.";
                            std::ostringstream ss;
                            ss << "candidate(" << candidate << ") != incumbent (" << incumbent << ")";
                            throw std::logic_error(ss.str());
                        }
                    } else {
                        ret[i] = candidate;
                    }
                }
            }
        }
    }
    return ret;
}

template<class DataKey, class ParameterKey>
torch::OrderedDict<DataKey, std::vector<int64_t>> reduce_sample_size_by_parameterisation_from_estimating_equations(
    const torch::OrderedDict<DataKey, torch::OrderedDict<ParameterKey, std::vector<torch::Tensor>>>& sample_size_to_reduce
) {
    torch::OrderedDict<DataKey, std::vector<int64_t>> ret; ret.reserve(sample_size_to_reduce.size());
    for (const auto& item : sample_size_to_reduce) {
        try {
            ret.insert(item.key(), reduce_sample_size_by_parameterisation_from_estimating_equations(item.value()));
        } catch(...) {
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << item.key() << " = series.";
            throw;
        }
    }
    return ret;
}

template<class DataKey>
torch::OrderedDict<DataKey, std::vector<int64_t>> reduce_sample_size_by_parameterisation_from_score(
    const torch::OrderedDict<DataKey, std::vector<torch::Tensor>> sample_size_to_reduce
) {
    torch::OrderedDict<DataKey, std::vector<int64_t>> ret; ret.reserve(sample_size_to_reduce.size());
    for (const auto& item : sample_size_to_reduce) {
        ret.insert(item.key(), reduce_sample_size_by_parameterisation(item.value()));
    }
    return ret;
}

template<class DataKey, class ParameterKey>
torch::OrderedDict<ParameterKey, torch::OrderedDict<ParameterKey, torch::Tensor>> truncated_kernel_asymptotic_covariance_matrix_panel(
    const torch::OrderedDict<DataKey, torch::OrderedDict<ParameterKey, torch::Tensor>>& x,
    const torch::OrderedDict<ParameterKey, torch::Tensor>& sample_size_by_parameter,
    const torch::OrderedDict<DataKey, std::vector<std::vector<torch::indexing::TensorIndex>>>& observations_by_parameterisation,
    const torch::OrderedDict<DataKey, torch::OrderedDict<ParameterKey, std::vector<torch::Tensor>>>& sample_size_by_parameterisation_unreduced,
    int64_t t_dim = -1,
    double na_cov = missing::na
) {
    auto sample_size_by_parameterisation = reduce_sample_size_by_parameterisation_from_estimating_equations(sample_size_by_parameterisation_unreduced);

    torch::OrderedDict<ParameterKey, torch::Tensor> sample_size_by_parameter_sqrt_rec;
    sample_size_by_parameter_sqrt_rec.reserve(sample_size_by_parameter.size());
    for (const auto& item : sample_size_by_parameter) {
        sample_size_by_parameter_sqrt_rec.insert(item.key(), item.value().toType(torch::kDouble).sqrt().reciprocal());
    }

    torch::OrderedDict<ParameterKey, torch::OrderedDict<ParameterKey, torch::Tensor>> sample_size_adjustment_each_parameter;
    sample_size_adjustment_each_parameter.reserve(sample_size_by_parameter.size());
    for (const auto& item_i : sample_size_by_parameter_sqrt_rec) {
        const auto& key_i = item_i.key();
        const auto& ssbp_i = item_i.value();
        if (ssbp_i.ndimension() != 1) { throw std::logic_error("ssbp_i.ndimension() != 1"); }
        auto ssbp_i_col_vec = ssbp_i.unsqueeze(1);
        torch::OrderedDict<ParameterKey, torch::Tensor> ssaep_i; ssaep_i.reserve(sample_size_by_parameter.size());
        for (const auto& item_j : sample_size_by_parameter_sqrt_rec) {
            const auto& key_j = item_j.key();
            const auto& ssbp_j = item_j.value();
            auto ssbp_j_row_vec = ssbp_j.unsqueeze(0);
            ssaep_i.insert(key_j, torch::matmul(ssbp_i_col_vec, ssbp_j_row_vec));
        }
        sample_size_adjustment_each_parameter.insert(key_i, std::move(ssaep_i));
    }

    torch::OrderedDict<ParameterKey, torch::OrderedDict<ParameterKey, torch::Tensor>> out; out.reserve(sample_size_by_parameter.size());
    for (const auto& item : x) {
        const auto& data_key = item.key();
        const auto& xi = item.value();
        const auto& ssbp_i = sample_size_by_parameterisation[data_key];
        const auto& obp_i = observations_by_parameterisation[data_key];
        if (ssbp_i.size() != obp_i.size()) { throw std::logic_error("ssbp_i.size() != obp_i.size()"); }
        for (int64_t j = 0; j != ssbp_i.size(); ++j) {
            auto sample_size_adjustment = ssbp_i[j]*sample_size_adjustment_each_parameter;
            if (out.is_empty()) {
                out = sample_size_adjustment*truncated_kernel_asymptotic_covariance_matrix(xi, obp_i[j], t_dim, na_cov);
            } else {
                out += sample_size_adjustment*truncated_kernel_asymptotic_covariance_matrix(xi, obp_i[j], t_dim, na_cov);
            }
        }
    }

    return out;
}

#endif

