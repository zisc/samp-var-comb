#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <R.h>
#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <torch/torch.h>
#include <modelling/fit.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/score/ScoringRule.hpp>
#include <R_modelling/fit.hpp>

SEXP fit_diagnostics_to_R_list(
    const FitDiagnostics& diagnostics,
    R_protect_guard& protect_guard
) { return R_handle_exception([&](){
    auto niterations = diagnostics.score.size();
    auto nparameters = diagnostics.parameters.at(0).size();
    auto ngrad = diagnostics.gradient.at(0).size();

    SEXP diagnostics_R = protect_guard.protect(Rf_allocVector(VECSXP, 4));
    
    SEXP diagnostics_R_names = protect_guard.protect(Rf_allocVector(STRSXP, 4));
    SET_STRING_ELT(diagnostics_R_names, 0, Rf_mkChar("seconds"));
    SET_STRING_ELT(diagnostics_R_names, 1, Rf_mkChar("score"));
    SET_STRING_ELT(diagnostics_R_names, 2, Rf_mkChar("parameters"));
    SET_STRING_ELT(diagnostics_R_names, 3, Rf_mkChar("gradient"));
    Rf_setAttrib(diagnostics_R, R_NamesSymbol, diagnostics_R_names);

    SEXP diagnostics_R_seconds = Rf_allocVector(INTSXP, 1);
    SET_VECTOR_ELT(diagnostics_R, 0, diagnostics_R_seconds);
    INTEGER(diagnostics_R_seconds)[0] = diagnostics.seconds;

    SEXP diagnostics_R_score = Rf_allocVector(REALSXP, niterations);
    SET_VECTOR_ELT(diagnostics_R, 1, diagnostics_R_score);
    std::memcpy(REAL(diagnostics_R_score), diagnostics.score.data(), sizeof(double)*niterations);

    SEXP diagnostics_R_parameters = Rf_allocVector(VECSXP, nparameters);
    SET_VECTOR_ELT(diagnostics_R, 2, diagnostics_R_parameters);
    SEXP diagnostics_R_parameters_names = protect_guard.protect(Rf_allocVector(STRSXP, nparameters));
    std::vector<std::vector<double*>> diagnostics_R_parameters_ptr;
    diagnostics_R_parameters_ptr.reserve(nparameters);

    SEXP diagnostics_R_gradient = Rf_allocVector(VECSXP, ngrad);
    SET_VECTOR_ELT(diagnostics_R, 3, diagnostics_R_gradient);
    SEXP diagnostics_R_gradient_names = protect_guard.protect(Rf_allocVector(STRSXP, ngrad));
    std::vector<std::vector<double*>> diagnostics_R_gradient_ptr;
    diagnostics_R_gradient_ptr.reserve(ngrad);

    for (decltype(nparameters) i = 0; i != nparameters; ++i) {
        auto param_i = diagnostics.parameters.at(0)[i];
        auto param_i_name = param_i.key();
        SET_STRING_ELT(diagnostics_R_parameters_names, i, Rf_mkChar(param_i_name.c_str()));
        auto param_i_value = param_i.value();
        auto param_size = param_i_value.numel();
        std::vector<double*> diagnostics_R_parameters_ptr_i;
        diagnostics_R_parameters_ptr_i.reserve(param_size);
        SEXP diagnostics_R_parameters_i = Rf_allocVector(VECSXP, param_size);
        SET_VECTOR_ELT(diagnostics_R_parameters, i, diagnostics_R_parameters_i);
        for (decltype(param_size) j = 0; j != param_size; ++j) {
            SEXP diagnostics_R_parameters_i_j = Rf_allocVector(REALSXP, niterations);
            SET_VECTOR_ELT(diagnostics_R_parameters_i, j, diagnostics_R_parameters_i_j);
            diagnostics_R_parameters_ptr_i.emplace_back(REAL(diagnostics_R_parameters_i_j));
        }
        diagnostics_R_parameters_ptr.emplace_back(std::move(diagnostics_R_parameters_ptr_i));
    }

    for (decltype(ngrad) i = 0; i != ngrad; ++i) {
        auto grad_i = diagnostics.gradient.at(0)[i];
        auto grad_i_name = grad_i.key();
        SET_STRING_ELT(diagnostics_R_gradient_names, i, Rf_mkChar(grad_i_name.c_str()));
        auto grad_i_value = grad_i.value();
        auto grad_size = grad_i_value.numel();
        std::vector<double*> diagnostics_R_gradient_ptr_i;
        diagnostics_R_gradient_ptr_i.reserve(grad_size);
        SEXP diagnostics_R_gradient_i = Rf_allocVector(VECSXP, grad_size);
        SET_VECTOR_ELT(diagnostics_R_gradient, i, diagnostics_R_gradient_i);
        for (decltype(grad_size) j = 0; j != grad_size; ++j) {
            SEXP diagnostics_R_gradient_i_j = Rf_allocVector(REALSXP, niterations);
            SET_VECTOR_ELT(diagnostics_R_gradient_i, j, diagnostics_R_gradient_i_j);
            diagnostics_R_gradient_ptr_i.emplace_back(REAL(diagnostics_R_gradient_i_j));
        }
        diagnostics_R_gradient_ptr.emplace_back(std::move(diagnostics_R_gradient_ptr_i));
    }
    
    for (decltype(niterations) i = 0; i != niterations; ++i) {
        auto params_i = diagnostics.parameters.at(i);
        for (decltype(nparameters) j = 0; j != nparameters; ++j) {
            auto params_i_j = params_i[j];
            auto params_i_j_value = params_i_j.value();
            auto params_i_j_value_ptr = params_i_j_value.data_ptr<double>();
            for (int64_t k = 0; k != params_i_j_value.numel(); ++k) {
                diagnostics_R_parameters_ptr[j][k][i] = params_i_j_value_ptr[k];
            }
        }

        auto grad_i = diagnostics.gradient.at(i);
        for (decltype(ngrad) j = 0; j != ngrad; ++j) {
            auto grad_i_j = grad_i[j];
            auto grad_i_j_value = grad_i_j.value();
            auto grad_i_j_value_ptr = grad_i_j_value.data_ptr<double>();
            for (int64_t k = 0; k != grad_i_j_value.numel(); ++k) {
                diagnostics_R_gradient_ptr[j][k][i] = grad_i_j_value_ptr[k];
            }
        }
    }

    Rf_setAttrib(diagnostics_R_parameters, R_NamesSymbol, diagnostics_R_parameters_names);
    Rf_setAttrib(diagnostics_R_gradient, R_NamesSymbol, diagnostics_R_gradient_names);

    return diagnostics_R;
});}

SEXP R_fit(
    SEXP models_R,
    SEXP scoring_rule_R,
    SEXP data_dict_R,
    SEXP learning_rate_R,
    SEXP barrier_begin_R,
    SEXP barrier_end_R,
    SEXP barrier_decay_R,
    SEXP tolerance_grad_R,
    SEXP tolerance_change_R,
    SEXP maximum_optimiser_iterations_R,
    SEXP timeout_in_seconds_R,
    SEXP return_diagnostics_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;

    auto scoring_rule = EXTPTRSXP_to_shared_ptr<ScoringRule>(scoring_rule_R);
    auto data = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(data_dict_R);
    double learning_rate = REAL(learning_rate_R)[0];
    double barrier_begin = REAL(barrier_begin_R)[0];
    double barrier_end = REAL(barrier_end_R)[0];
    double barrier_decay = REAL(barrier_decay_R)[0];
    double tolerance_grad = REAL(tolerance_grad_R)[0];
    double tolerance_change = REAL(tolerance_change_R)[0];
    int maximum_optimiser_iterations = INTEGER(maximum_optimiser_iterations_R)[0];
    int timeout_in_seconds = INTEGER(timeout_in_seconds_R)[0];
    int return_diagnostics = LOGICAL(return_diagnostics_R)[0];

    int64_t nmodels = Rf_length(models_R);

    SEXP ret_R = protect_guard.protect(Rf_allocVector(VECSXP, 3));
    SEXP ret_R_names = Rf_allocVector(STRSXP, 3);
    Rf_setAttrib(ret_R, R_NamesSymbol, ret_R_names);

    SEXP fit_models_R = Rf_allocVector(VECSXP, nmodels);
    SET_VECTOR_ELT(ret_R, 0, fit_models_R);
    SET_STRING_ELT(ret_R_names, 0, Rf_mkChar("models"));

    SEXP success_R = Rf_allocVector(VECSXP, nmodels);
    SET_VECTOR_ELT(ret_R, 1, success_R);
    SET_STRING_ELT(ret_R_names, 1, Rf_mkChar("success"));

    SEXP diagnostics_R = R_NilValue;
    if (return_diagnostics) diagnostics_R = Rf_allocVector(VECSXP, nmodels);
    SET_VECTOR_ELT(ret_R, 2, diagnostics_R);
    SET_STRING_ELT(ret_R_names, 2, Rf_mkChar("diagnostics"));

    torch::optim::LBFGSOptions lbfgs_options(learning_rate);
    lbfgs_options.line_search_fn("strong_wolfe");
    lbfgs_options.tolerance_grad(tolerance_grad);
    lbfgs_options.tolerance_change(tolerance_change);

    auto fit_i = [&](int64_t i, const auto& model, FitDiagnostics *diagnostics_ptr) {
        bool success;
        SET_VECTOR_ELT(
            fit_models_R,
            i,
            shared_ptr_to_EXTPTRSXP<ProbabilisticModule, torch::nn::Module>(
                fit(
                    model,
                    data,
                    scoring_rule,
                    barrier_begin,
                    barrier_end,
                    barrier_decay,
                    maximum_optimiser_iterations,
                    timeout_in_seconds,
                    lbfgs_options,
                    diagnostics_ptr,
                    &success
                ),
                protect_guard
            )
        );
        SEXP success_R_i = Rf_allocVector(LGLSXP, 1);
        LOGICAL(success_R_i)[0] = success;
        SET_VECTOR_ELT(success_R, i, success_R_i);
    };

    for (int64_t i = 0; i != nmodels; ++i) {
        auto model = EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(VECTOR_ELT(models_R, i));
        if (return_diagnostics) {
            FitDiagnostics diagnostics;
            fit_i(i, model, &diagnostics);
            SET_VECTOR_ELT(
                diagnostics_R,
                i,
                fit_diagnostics_to_R_list(
                    diagnostics,
                    protect_guard
                )
            );
        } else {
            fit_i(i, model, nullptr);
        }
    }

    return ret_R;
});}

