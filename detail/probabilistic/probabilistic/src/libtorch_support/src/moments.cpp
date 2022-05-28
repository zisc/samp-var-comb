#include <vector>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <libtorch_support/moments.hpp>

#include <log/trivial.hpp>

torch::Tensor sample_size_present(
    const torch::Tensor& x_present
) {
    return x_present.sum();
}

torch::Tensor sample_size(
    const torch::Tensor& x,
    double na_ss
) {
    return sample_size_present(missing::is_present(x, na_ss));
}

torch::Tensor sum_present(
    const torch::Tensor& x,
    const torch::Tensor& x_present
) {
    auto x_selected_present = x.masked_select(x_present);
    return x_selected_present.numel() ? x_selected_present.sum() : x.new_zeros({});
}

torch::Tensor sum(
    const torch::Tensor& x,
    double na_s
) {
    return sum_present(x, missing::is_present(x, na_s));
}

torch::Tensor average_present(
    const torch::Tensor& x,
    const torch::Tensor& x_present
) {
    return sum_present(x, x_present)/sample_size_present(x_present);
}

torch::Tensor average(
    const torch::Tensor& x,
    double na_av
) {
    if (!x.numel()) {
        return torch::empty({}, torch::kDouble);
    }

    auto x_present = missing::is_present(x, na_av);
    
    if (!x_present.any().item<bool>()) {
        return x.new_full({}, na_av);
    }

    return average_present(x, x_present);
}

torch::Tensor average_once_zeroed(
    const torch::Tensor& x_zeroed,
    const std::vector<int64_t>& dim,
    torch::Tensor x_sample_sizes,
    bool keepdim
) {
    return x_zeroed.sum(dim, keepdim)/x_sample_sizes;
}

torch::Tensor average(
    const torch::Tensor& x,
    const std::vector<int64_t>& dim,
    bool keepdim,
    double na_av
) {
    auto x_present = x.ne(na_av);
    auto x_sample_sizes = x_present.sum(dim, keepdim, torch::kLong);
    return average_once_zeroed(
        missing::replace_na(x, na_av, 0.0),
        dim,
        x_sample_sizes,
        keepdim
    );
}

// For cross_covariance_matrix, assume that the last dimension
// in the index identifies the element of a vector which is
// an observation, and the prior dimensions of the index identify
// the observation.

void get_dimensions_across_observations_impl(
    std::vector<int64_t>& dimensions_across_observations,
    int64_t ndimension
) {
    dimensions_across_observations.reserve(ndimension-1);
    for (int64_t i = 0; i != ndimension-1; ++i) {
        dimensions_across_observations.emplace_back(i);
    }
}

void get_dimensions_across_observations(
    std::vector<int64_t>& dimensions_across_observations,
    int64_t ndimension
) {
    if (dimensions_across_observations.empty()) {
        get_dimensions_across_observations_impl(
            dimensions_across_observations,
            ndimension
        );
    }
}

void get_dimensions_across_observations(
    std::vector<int64_t>& dimensions_across_observations,
    const torch::Tensor& x
) {
    if (dimensions_across_observations.empty()) {
        get_dimensions_across_observations_impl(
            dimensions_across_observations,
            x.ndimension()
        );
    }
}

torch::Tensor cross_covariance_matrix_once_zeroed(
    const torch::Tensor& outer_products_zeroed,
    const torch::Tensor& outer_products_sample_sizes,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
) {
    auto zero_sample_size = outer_products_sample_sizes.eq(0);
    auto positive_sample_size = zero_sample_size.logical_not();
    auto inv_outer_products_sample_sizes = torch::empty(outer_products_sample_sizes.sizes(), torch::kDouble);
    inv_outer_products_sample_sizes.index_put_({zero_sample_size}, 0.0);
    inv_outer_products_sample_sizes.index_put_(
        {positive_sample_size},
        outer_products_sample_sizes.masked_select(positive_sample_size)
                                   .toType(torch::kDouble)
                                   .reciprocal()
    );
    auto outer_products_zeroed_average = outer_products_zeroed.sum(dimensions_across_observations)*inv_outer_products_sample_sizes;
    outer_products_zeroed_average.index_put_({zero_sample_size}, na_cov);
    return outer_products_zeroed_average;
}

torch::Tensor cross_covariance_matrix_once_zeroed(
    const torch::Tensor& x_centered_zeroed,
    const torch::Tensor& y_centered_zeroed,
    const torch::Tensor& outer_products_sample_sizes,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
) {
    auto ndimension = x_centered_zeroed.ndimension();

    auto outer_products_zeroed = torch::matmul(
        x_centered_zeroed.unsqueeze(ndimension),
        y_centered_zeroed.unsqueeze(ndimension-1)
    );

    return cross_covariance_matrix_once_zeroed(
        outer_products_zeroed,
        outer_products_sample_sizes,
        na_cov,
        dimensions_across_observations
    );
}

torch::Tensor cross_covariance_matrix_once_zeroed(
    const torch::Tensor& x_centered_zeroed,
    const torch::Tensor& y_centered_zeroed,
    const torch::Tensor& outer_product_sample_sizes,
    double na_cov,
    std::vector<int64_t>&& dimensions_across_observations
) {
    get_dimensions_across_observations(dimensions_across_observations, x_centered_zeroed);
    return cross_covariance_matrix_once_zeroed(
        x_centered_zeroed,
        y_centered_zeroed,
        outer_product_sample_sizes,
        na_cov,
        dimensions_across_observations
    );
}

torch::Tensor cross_covariance_matrix_once_centered(
    const torch::Tensor& x_centered,
    const torch::Tensor& y_centered,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
) {
    auto ndimension = x_centered.ndimension();

    auto x_present = x_centered.ne(na_cov).toType(torch::kLong);
    auto y_present = y_centered.ne(na_cov).toType(torch::kLong);
    auto outer_products_present = torch::matmul(
        x_present.unsqueeze(ndimension),
        y_present.unsqueeze(ndimension-1)
    );
    auto outer_products_sample_sizes = outer_products_present.sum(dimensions_across_observations);

    return cross_covariance_matrix_once_zeroed(
        missing::replace_na(x_centered, na_cov, 0.0),
        missing::replace_na(y_centered, na_cov, 0.0),
        outer_products_sample_sizes,
        na_cov,
        dimensions_across_observations
    );
}

torch::Tensor cross_covariance_matrix_once_centered(
    const torch::Tensor& x_centered,
    const torch::Tensor& y_centered,
    double na_cov,
    std::vector<int64_t>&& dimensions_across_observations
) {
    get_dimensions_across_observations(dimensions_across_observations, x_centered);
    return cross_covariance_matrix_once_centered(x_centered, y_centered, na_cov, dimensions_across_observations);
}

torch::Tensor cross_covariance_matrix(
    const torch::Tensor& x,
    const torch::Tensor& x_average,
    const torch::Tensor& y,
    const torch::Tensor& y_average,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
) {
    auto x_centered = x - x_average;
    x_centered.index_put_({x.eq(na_cov)}, na_cov);

    auto y_centered = y - y_average;
    y_centered.index_put_({y.eq(na_cov)}, na_cov);

    return cross_covariance_matrix_once_centered(
        x_centered,
        y_centered,
        na_cov,
        dimensions_across_observations
    );
}

torch::Tensor cross_covariance_matrix(
    const torch::Tensor& x,
    const torch::Tensor& x_average,
    const torch::Tensor& y,
    const torch::Tensor& y_average,
    double na_cov,
    std::vector<int64_t>&& dimensions_across_observations
) {
    get_dimensions_across_observations(dimensions_across_observations, x);
    return cross_covariance_matrix(x, x_average, y, y_average, na_cov, dimensions_across_observations);
}

torch::Tensor cross_covariance_matrix(
    const torch::Tensor& x,
    const torch::Tensor& y,
    double na_cov,
    const std::vector<int64_t>& dimensions_across_observations
) {
    return cross_covariance_matrix(
        x,
        average(x, dimensions_across_observations, true, na_cov),
        y,
        average(y, dimensions_across_observations, true, na_cov),
        na_cov,
        dimensions_across_observations
    );
}

torch::Tensor cross_covariance_matrix(
    const torch::Tensor& x,
    const torch::Tensor& y,
    double na_cov,
    std::vector<int64_t>&& dimensions_across_observations
) {
    get_dimensions_across_observations(dimensions_across_observations, x);
    return cross_covariance_matrix(x, y, na_cov, dimensions_across_observations);
}

int64_t reduce_sample_size_by_parameterisation(
    const torch::Tensor& sample_size_to_reduce
) {
    auto sample_size_to_reduce_long = sample_size_to_reduce.toType(torch::kLong);
    auto one_true_sample_size = sample_size_to_reduce_long.max();
    
    #ifndef NDEBUG
        auto mask = sample_size_to_reduce_long.ne(one_true_sample_size);
        if (
            static_cast<torch::Tensor>(mask.any()).item<bool>() &&
            static_cast<torch::Tensor>(sample_size_to_reduce_long.masked_select({mask}).max()).item<int64_t>() != 0
        ) {
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << sample_size_to_reduce << " = sample_size_to_reduce\n";
            throw std::logic_error("sample_size_to_reduce has more than one non-zero sample size.");
        }
    #endif

    return one_true_sample_size.item<int64_t>();
}

std::vector<int64_t> reduce_sample_size_by_parameterisation(
    const std::vector<torch::Tensor>& sample_size_to_reduce
) {
    std::vector<int64_t> out; out.reserve(sample_size_to_reduce.size());
    int64_t i = 0;
    for (const auto& ss : sample_size_to_reduce) {
        out.emplace_back(reduce_sample_size_by_parameterisation(ss));
    }
    return out;
}

