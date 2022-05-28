#ifndef PROBABILISTIC_MODELLING_DISTRIBUTION_NORMAL_VECTOR_HPP_GUARD
#define PROBABILISTIC_MODELLING_DISTRIBUTION_NORMAL_VECTOR_HPP_GUARD

#include <memory>
#include <string>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>

/*
std::unique_ptr<Distribution> ManufactureNormalVector(
    torch::OrderedDict<std::string, torch::Tensor> mean,
    torch::OrderedDict<std::pair<std::string, std::string>, torch::Tensor> variance
);
*/

std::unique_ptr<Distribution> ManufactureNormalVectorDetail(
    torch::Tensor mu,
    torch::Tensor A,
    torch::OrderedDict<std::string, torch::indexing::TensorIndex> indices
);

#endif

