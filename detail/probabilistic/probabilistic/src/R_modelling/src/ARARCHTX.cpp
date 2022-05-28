#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_modelling/model/manufacture.hpp>
#include <modelling/model/ARARCHTX.hpp>
#include <R_modelling/model/ARARCHTX.hpp>

SEXP R_ManufactureARARCHTX(
    SEXP parameter_guesses_R,
    SEXP buffers_R,
    SEXP observations_dict_R,
    SEXP pre_optimise_R
) { return R_handle_exception([&]() {
    auto pre_optimise = LOGICAL(pre_optimise_R)[0];
    if (pre_optimise) {
        return R_ManufactureProbabilisticModule(
            [](
                NamedShapelyParameters& parameters,
                Buffers& buffers,
                const torch::OrderedDict<std::string, torch::Tensor>& observations
            ) {
                return ManufactureARARCHTX(parameters, buffers, observations);
            },
            parameter_guesses_R,
            buffers_R,
            observations_dict_R
        );
    } else {
        return R_ManufactureProbabilisticModule(
            [](
                NamedShapelyParameters& parameters,
                Buffers& buffers,
                const torch::OrderedDict<std::string, torch::Tensor>& observations
            ) {
                return ManufactureARARCHTX(parameters, buffers);
            },
            parameter_guesses_R,
            buffers_R,
            observations_dict_R
        );
    }
});}

