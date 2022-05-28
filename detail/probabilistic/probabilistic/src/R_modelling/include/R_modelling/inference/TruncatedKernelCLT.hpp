#ifndef PROBABILISTIC_R_MODELLING_TRUNCATED_KERNEL_CLT_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_TRUNCATED_KERNEL_CLT_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_ManufactureTruncatedKernelCLT(
        SEXP fit_R,
        SEXP dependent_index_R
    );
}

#endif

