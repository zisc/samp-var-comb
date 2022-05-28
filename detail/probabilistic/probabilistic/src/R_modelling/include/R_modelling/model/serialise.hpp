#ifndef PROBABILISTIC_R_MODELLING_SERIALISE_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_SERIALISE_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_serialise_model(SEXP models_R);
    DLL_PUBLIC SEXP R_deserialise_model(SEXP models_out_R, SEXP models_serialised_R);
}

#endif

