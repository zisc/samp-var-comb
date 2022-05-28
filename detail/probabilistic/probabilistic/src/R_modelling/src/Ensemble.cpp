#include <memory>
#include <vector>
#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_modelling/buffers.hpp>
#include <R_modelling/parameterisation.hpp>
#include <modelling/model/Ensemble.hpp>
#include <R_modelling/model/Ensemble.hpp>

inline std::vector<std::shared_ptr<ProbabilisticModule>> get_components(SEXP components_R) {
    auto num_components = Rf_length(components_R);
    std::vector<std::shared_ptr<ProbabilisticModule>> components; components.reserve(num_components);
    for (decltype(num_components) i = 0; i != num_components; ++i) {
        components.emplace_back(EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(VECTOR_ELT(components_R, i)));
    }
    return components;
}

SEXP R_ManufactureEnsemble(
    SEXP components_R,
    SEXP parameter_guesses_R,
    SEXP buffers_R
) {
    R_protect_guard protect_guard;

    auto components = get_components(components_R);

    auto buffers = to_buffers(buffers_R);

    auto nguesses = Rf_length(parameter_guesses_R);
    SEXP modules = protect_guard.protect(Rf_allocVector(VECSXP, nguesses));
    for (decltype(nguesses) i = 0; i != nguesses; ++i) {
        auto shapely_parameters = to_named_shapely_parameters(VECTOR_ELT(parameter_guesses_R, i));
        auto buffers_clone = buffers.clone();
        SET_VECTOR_ELT(modules, i, shared_ptr_to_EXTPTRSXP<ProbabilisticModule, torch::nn::Module>(
            ManufactureEnsemble(
                components,
                shapely_parameters,
                buffers_clone
            ),
            protect_guard
        ));
    }

    return modules;
}

SEXP R_change_components(
    SEXP ensemble_R,
    SEXP new_components_R
) {
    R_protect_guard protect_guard;
    auto ensemble = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(ensemble_R);
    auto new_components = get_components(new_components_R);
    return shared_ptr_to_EXTPTRSXP<ProbabilisticModule, torch::nn::Module>(
        change_components(*ensemble, new_components),
        protect_guard
    );
}

