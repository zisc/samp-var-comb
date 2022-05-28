#include <sstream>
#include <stdexcept>
#include <utility>
#include <torch/torch.h>
#include <Rinternals.h>
#include <R_modelling/parameterisation.hpp>

ShapelyParameter to_shapely_parameter(
    SEXP shapely_parameter_R
) {
    auto to_double = [](SEXP in, const char *name) -> double {
        if (!Rf_isReal(in)) {
            std::ostringstream ss;
            ss << "to_shapely_parameter: !Rf_isReal(" << name << ")";
            throw std::logic_error(ss.str());
        }
        if (Rf_length(in) != 1) {
            std::ostringstream ss;
            ss << "to_shapely_parameter: Rf_length(" << name << ") != 1";
            throw std::logic_error(ss.str());
        }

        return REAL(in)[0];
    };

    auto to_bool = [](SEXP in, const char *name) -> bool {
        if (!Rf_isLogical(in)) {
            std::ostringstream ss;
            ss << "to_shapely_parameter: !Rf_isLogical(" << name << ")";
            throw std::logic_error(ss.str());
        }
        if (Rf_length(in) != 1) {
            std::ostringstream ss;
            ss << "to_shapely_parameter: Rf_length(" << name << ") != 1";
            throw std::logic_error(ss.str());
        }

        return INTEGER(in)[0];
    };

    if (!Rf_isNewList(shapely_parameter_R)) {
        throw std::logic_error("to_shapely_parameter: !Rf_isNewList(shapely_parameter_R)");
    }
    if (Rf_length(shapely_parameter_R) != 6) {
        throw std::logic_error("to_shapely_parameter: Rf_length(shapely_parameter_R) != 6");
    }

    SEXP parameter_R = VECTOR_ELT(shapely_parameter_R, 0);
    if (!Rf_isReal(parameter_R)) {
        throw std::logic_error("to_shapely_parameter: !Rf_isReal(parameter_R)");
    }

    double *parameter = REAL(parameter_R);
    auto parameter_size = Rf_length(parameter_R);
    torch::Tensor parameter_tensor = torch::empty({parameter_size}, torch::kDouble);
    auto parameter_tensor_a = parameter_tensor.accessor<double,1>();
    for (decltype(parameter_size) i = 0; i != parameter_size; ++i) {
        parameter_tensor_a[i] = parameter[i];
    }

    auto lower_bound = to_double(VECTOR_ELT(shapely_parameter_R, 1), "lower_bound");
    auto upper_bound = to_double(VECTOR_ELT(shapely_parameter_R, 2), "upper_bound");
    auto parameter_scaling = to_double(VECTOR_ELT(shapely_parameter_R, 3), "parameter_scaling");
    auto barrier_scaling = to_double(VECTOR_ELT(shapely_parameter_R, 4), "barrier_scaling");
    auto enable = to_bool(VECTOR_ELT(shapely_parameter_R, 5), "enable");

    ShapelyParameter out = {
        std::move(parameter_tensor),
        lower_bound,
        upper_bound,
        parameter_scaling,
        barrier_scaling,
        enable
    };

    return out;
}

NamedShapelyParameters to_named_shapely_parameters(
    SEXP named_shapely_parameter_list_R
) {
    if (!Rf_isNewList(named_shapely_parameter_list_R)) {
        throw std::logic_error("to_named_shapely_parameters: !Rf_isNewList(named_shapely_parameter_list_R)");
    }

    SEXP parameters_R = VECTOR_ELT(named_shapely_parameter_list_R, 0);
    if (!Rf_isNewList(parameters_R)) {
        throw std::logic_error("to_named_shapely_parameters: !Rf_isNewList(parameters_R)");
    }

    SEXP parameter_names_R = Rf_getAttrib(named_shapely_parameter_list_R, R_NamesSymbol);
    auto nparams = Rf_length(named_shapely_parameter_list_R);

    torch::OrderedDict<std::string, ShapelyParameter> parameters;
    parameters.reserve(nparams);

    for (decltype(nparams) i = 0; i != nparams; ++i) {
        auto param_name = CHAR(STRING_ELT(parameter_names_R, i));
        parameters.insert(
            param_name,
            to_shapely_parameter(VECTOR_ELT(named_shapely_parameter_list_R, i))
        );
    }

    NamedShapelyParameters out = {std::move(parameters)};

    return out;
}

