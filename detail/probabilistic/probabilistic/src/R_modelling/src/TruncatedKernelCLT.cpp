#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/inference/SamplingDistribution.hpp>
#include <modelling/inference/TruncatedKernelCLT.hpp>
#include <R_modelling/inference/TruncatedKernelCLT.hpp>

SEXP R_ManufactureTruncatedKernelCLT(
    SEXP fit_R,
    SEXP dependent_index_R
) { return R_handle_exception([&]() {
    R_protect_guard protect_guard;
    return shared_ptr_to_EXTPTRSXP(
        ManufactureTruncatedKernelCLT(
            EXTPTRSXP_to_shared_ptr<ProbabilisticModule, torch::nn::Module>(fit_R),
            INTEGER(dependent_index_R)[0]
        ),
        protect_guard
    );
});}

