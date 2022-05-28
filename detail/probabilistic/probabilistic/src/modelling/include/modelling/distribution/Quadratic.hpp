#ifndef PROBABILISTIC_MODELLING_DISTRIBUTION_QUADRATIC_HPP_GUARD
#define PROBABILISTIC_MODELLING_DISTRIBUTION_QUADRATIC_HPP_GUARD

#include <memory>
#include <string>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>

std::unique_ptr<Distribution> ManufactureQuadratic(
    std::string name,
    std::shared_ptr<Distribution> dist,
    double intercept = 0.0,
    torch::OrderedDict<std::string, torch::Tensor> linear_coefficients = torch::OrderedDict<std::string, torch::Tensor>(),
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> quadratic_coefficients = torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>>()
);

std::unique_ptr<Distribution> ManufactureQuadratic(
    std::string name,
    std::shared_ptr<Distribution> dist,
    torch::Tensor intercept,
    torch::OrderedDict<std::string, torch::Tensor> linear_coefficients = torch::OrderedDict<std::string, torch::Tensor>(),
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> quadratic_coefficients = torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>>()
);

#endif

