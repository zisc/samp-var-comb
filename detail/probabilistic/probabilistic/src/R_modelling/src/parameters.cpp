#include <cstring>
#include <log/trivial.hpp>
#include <R.h>
#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <modelling/model/ProbabilisticModule.hpp>
#include <R_modelling/model/parameters.hpp>

SEXP R_parameters(
    SEXP models_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;

    auto nmodels = Rf_length(models_R);

    SEXP parameters_R = protect_guard.protect(Rf_allocVector(VECSXP, nmodels));

    for (decltype(nmodels) i = 0; i != nmodels; ++i) {
        SEXP models_R_i = VECTOR_ELT(models_R, i);

        auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(models_R_i);

        auto parameters = model->named_parameters_on_paper();
        auto nparameters = parameters.size();
        SEXP parameters_R_i = Rf_allocVector(VECSXP, nparameters);
        SET_VECTOR_ELT(parameters_R, i, parameters_R_i);
        SEXP parameters_R_i_names = Rf_allocVector(STRSXP, nparameters);
        Rf_setAttrib(parameters_R_i, R_NamesSymbol, parameters_R_i_names);
        for (decltype(nparameters) j = 0; j != nparameters; ++j) {
            const auto& param_j = parameters[j];

            auto param_j_name = param_j.key();
            SET_STRING_ELT(parameters_R_i_names, j, Rf_mkChar(param_j_name.c_str()));

            auto param_j_value = param_j.value();
            auto param_j_value_numel = param_j_value.numel();
            auto param_j_value_ndimension = param_j_value.ndimension();
            auto param_j_value_sizes = param_j_value.sizes();

            SEXP parameters_R_ij = Rf_allocVector(VECSXP, 2);
            SET_VECTOR_ELT(parameters_R_i, j, parameters_R_ij);

            SEXP parameters_R_ij_data = Rf_allocVector(REALSXP, param_j_value_numel);
            SET_VECTOR_ELT(parameters_R_ij, 0, parameters_R_ij_data);
            double *parameters_R_ij_data_a = REAL(parameters_R_ij_data);
            std::memcpy(parameters_R_ij_data_a, param_j_value.data_ptr<double>(), sizeof(double)*param_j_value_numel);

            SEXP parameters_R_ij_dim = Rf_allocVector(INTSXP, param_j_value_ndimension);
            SET_VECTOR_ELT(parameters_R_ij, 1, parameters_R_ij_dim);
            int *parameters_R_ij_dim_a = INTEGER(parameters_R_ij_dim);
            for (int64_t k = 0; k != param_j_value_ndimension; ++k) {
                // Libtorch stores their tensors in row-major, but R stores their arrays in
                // param_i_value_ndimension-major. Copying the tensor data as-is into an
                // R array therefore copies the transpose, and the dimension sizes are reversed.
                // We will transpose back in the R caller of this function.
                parameters_R_ij_dim_a[k] = param_j_value_sizes[param_j_value_ndimension - k - 1];
            }

            SET_VECTOR_ELT(parameters_R_ij, 0, parameters_R_ij_data);
            SET_VECTOR_ELT(parameters_R_ij, 1, parameters_R_ij_dim);
            SET_VECTOR_ELT(parameters_R_i, j, parameters_R_ij);
        }
    }

    return parameters_R;
});}

DLL_PUBLIC SEXP R_change_parameters(
    SEXP model_to_change_R,
    SEXP model_with_parameters_R
) {
    R_protect_guard protect_guard;
    auto model_to_change = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(model_to_change_R);
    auto model_with_parameters = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(model_with_parameters_R);
    auto model_out = model_to_change->clone_probabilistic_module();
    model_out->set_parameters(model_with_parameters->named_parameters());
    return shared_ptr_to_EXTPTRSXP<ProbabilisticModule, torch::nn::Module>(model_out, protect_guard);
}

