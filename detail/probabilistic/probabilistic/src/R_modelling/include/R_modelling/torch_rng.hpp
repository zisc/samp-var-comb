#ifndef PROBABILISTIC_R_SEED_TORCH_RNG_HPP_GUARD
#define PROBABILISTIC_R_SEED_TORCH_RNG_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_seed_torch_rng(SEXP seed_R);
    DLL_PUBLIC SEXP R_get_state_torch_rng(void);
    DLL_PUBLIC SEXP R_set_state_torch_rng(SEXP state_R);
}

#endif

