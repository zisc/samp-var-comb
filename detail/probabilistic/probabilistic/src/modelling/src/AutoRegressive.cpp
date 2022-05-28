#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <libtorch_support/Parameterisation.hpp>
#include <modelling/model/ShapelyModule.hpp>
#include <modelling/model/AutoRegressive.hpp>

torch::Tensor AutoRegressive_forward(
    torch::Tensor x,
    torch::Tensor coefficients_get
) {
    auto x_sizes = x.sizes();
    auto t_size = x_sizes.back();

    auto ar_order = coefficients_get.sizes().back();

    std::vector<int64_t> ar_covariates_sizes;
    ar_covariates_sizes.reserve(x_sizes.size() + 1);
    for (const auto& s : x_sizes) {
        ar_covariates_sizes.emplace_back(s);
    }
    ar_covariates_sizes.emplace_back(ar_order);

    auto ar_covariates = x.new_full(ar_covariates_sizes, missing::na, torch::kDouble);
    for (decltype(ar_order) i = 0; i != ar_order; ++i) {
        ar_covariates.index_put_(
            {torch::indexing::Ellipsis, torch::indexing::Slice(i+1, t_size), i},
            x.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, t_size - i - 1)})
        );
    }

    return missing::handle_na(
        torch::matmul,
        ar_covariates,
        coefficients_get
    );
}

torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>> AutoRegressive_observations_by_parameter(
    const torch::Tensor& x,
    const std::shared_ptr<Parameterisation>& coefficients,
    bool recursive
) {
    torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>> out;
    out.insert(shapely_parameter_raw_name(*coefficients), {std::vector<torch::indexing::TensorIndex>(x.ndimension(), torch::indexing::Slice())});
    return out;
}

