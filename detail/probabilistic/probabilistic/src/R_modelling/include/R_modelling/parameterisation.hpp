#ifndef PROBABILISTIC_R_MODELLING_PARAMETERISATION_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_PARAMETERISATION_HPP_GUARD

#include <torch/torch.h>
#include <Rinternals.h>
#include <libtorch_support/Parameterisation.hpp>

ShapelyParameter to_shapely_parameter(
    SEXP shapely_parameter_R
);

/*
ParameterisationCharacteristics to_parameterisation_characteristics(
    SEXP parameterisation_characteristics_R
);
*/

torch::OrderedDict<std::string, ShapelyParameter> to_shapely_parameter_dictionary(
    SEXP shapely_parameter_list_R
);

NamedShapelyParameters to_named_shapely_parameters(
    SEXP named_shapely_parameter_list_R
);

#endif

