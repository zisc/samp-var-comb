#ifndef PROBABILISTIC_MODELLING_COMBINATION_HPP_GUARD
#define PROBABILISTIC_MODELLING_COMBINATION_HPP_GUARD

#include <memory>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/Parameterisation.hpp>
#include <modelling/model/ProbabilisticModule.hpp>

std::unique_ptr<ProbabilisticModule> ManufactureEnsemble(
    std::vector<std::shared_ptr<ProbabilisticModule>> components,
    NamedShapelyParameters& shapely_parameters,
    Buffers& buffers
);

std::shared_ptr<ProbabilisticModule> change_components(
    const ProbabilisticModule& ensemble,
    const std::vector<std::shared_ptr<ProbabilisticModule>>& new_components
);

#endif

