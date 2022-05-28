#ifndef R_MODELLING_MODEL_MANUFACTURE_HPP_GUARD
#define R_MODELLING_MODEL_MANUFACTURE_HPP_GUARD

#include <utility>
#include <Rinternals.h>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <libtorch_support/Parameterisation.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <R_modelling/parameterisation.hpp>
#include <R_modelling/buffers.hpp>

// For all R_ManufactureProbabilisticModule functions, the parameter
// "manufacture_module" must be a function or function object of the form:
// std::unique_ptr<ProbabilisticModule> manufacture_module(
//     NamedShapelyParameters& parameters,
//     Buffers& buffers,
//     const torch::OrderedDict<std::string, torch::Tensor>& observations
// ).

template<class F>
SEXP R_ManufactureProbabilisticModule(
    F&& manufacture_module,
    SEXP parameter_guesses_R,
    SEXP buffers_R,
    SEXP observations_dict_R,
    R_protect_guard& protect_guard
) {
    auto observations = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(observations_dict_R);
    auto buffers = to_buffers(buffers_R);
    int64_t nguesses = Rf_length(parameter_guesses_R);
    SEXP modules = protect_guard.protect(Rf_allocVector(VECSXP, nguesses));
    for (int64_t i = 0; i != nguesses; ++i) {
        auto shapely_parameters = to_named_shapely_parameters(VECTOR_ELT(parameter_guesses_R, i));
        auto buffers_clone = buffers.clone();
        SET_VECTOR_ELT(modules, i, shared_ptr_to_EXTPTRSXP<ProbabilisticModule, torch::nn::Module>(
            manufacture_module(
                shapely_parameters,
                buffers_clone,
                *observations
            ),
            protect_guard
        ));
    }
    return modules;
}

template<class F>
SEXP R_ManufactureProbabilisticModule(
    F&& manufacture_module,
    SEXP parameter_guesses_R,
    SEXP buffers_R,
    SEXP observations_dict_R
) {
    R_protect_guard protect_guard;
    return R_ManufactureProbabilisticModule(
        std::forward<F>(manufacture_module),
        parameter_guesses_R,
        buffers_R,
        observations_dict_R,
        protect_guard
    );
}

#endif

