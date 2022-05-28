#ifndef PROBABILISTIC_MODELLING_DISTRIBUTION_MIXTURE_HPP_GUARD
#define PROBABILISTIC_MODELLING_DISTRIBUTION_MIXTURE_HPP_GUARD

#include <memory>
#include <vector>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>

std::unique_ptr<Distribution> ManufactureMixture(
    std::vector<std::shared_ptr<Distribution>> components,
    torch::Tensor weights
);

#endif

