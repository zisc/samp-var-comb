#ifndef PROBABILISTIC_MODELLING_ARARCHTX_HPP_GUARD
#define PROBABILISTIC_MODELLING_ARARCHTX_HPP_GUARD

#include <memory>
#include <torch/torch.h>
#include <libtorch_support/Parameterisation.hpp>
#include <modelling/model/ProbabilisticModule.hpp>

std::unique_ptr<ProbabilisticModule> ManufactureARARCHTX(
    NamedShapelyParameters& parameters,
    Buffers& buffers
);

std::unique_ptr<ProbabilisticModule> ManufactureARARCHTX(
    NamedShapelyParameters& parameters,
    Buffers& buffers,
    const torch::OrderedDict<std::string, torch::Tensor>& observations
);

#endif

