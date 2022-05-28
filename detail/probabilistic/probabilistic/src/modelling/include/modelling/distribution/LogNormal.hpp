#ifndef PROBABILISTIC_MODELLING_DISTRIBUTION_LOG_NORMAL_HPP_GUARD
#define PROBABILISTIC_MODELLING_DISTRIBUTION_LOG_NORMAL_HPP_GUARD

#include <memory>
#include <string>
#include <vector>
#include <torch/torch.h>

std::unique_ptr<Distribution> ManufactureLogNormal(
    torch::OrderedDict<std::string, torch::Tensor> mu,
    torch::OrderedDict<std::string, torch::Tensor> sigma
);

std::unique_ptr<Distribution> ManufactureLogNormal(const torch::OrderedDict<std::string, torch::Tensor>& tensors);

#endif

