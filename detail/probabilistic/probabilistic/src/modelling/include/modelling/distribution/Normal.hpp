#ifndef PROBABILISTIC_MODELLING_DISTRIBUTION_NORMAL_HPP_GUARD
#define PROBABILISTIC_MODELLING_DISTRIBUTION_NORMAL_HPP_GUARD

#include <memory>
#include <string>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>

std::unique_ptr<Distribution> ManufactureNormal(
    torch::OrderedDict<std::string, torch::Tensor> mean,
    torch::OrderedDict<std::string, torch::Tensor> std_dev
);

std::unique_ptr<Distribution> ManufactureNormal(const torch::OrderedDict<std::string, torch::Tensor>& tensors);

#endif

