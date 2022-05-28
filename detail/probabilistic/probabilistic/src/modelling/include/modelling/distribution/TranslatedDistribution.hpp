#ifndef PROBABILISTIC_MODELLING_DISTRIBUTION_TRANSLATED_HPP_GUARD
#define PROBABILISTIC_MODELLING_DISTRIBUTION_TRANSLATED_HPP_GUARD

#include <memory>
#include <string>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>

std::unique_ptr<Distribution> ManufactureTranslatedDistribution(
    std::shared_ptr<Distribution> distribution,
    const torch::OrderedDict<std::string, torch::Tensor>& translation
);

#endif

