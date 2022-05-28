#ifndef PROBABILISTIC_R_MODELLING_TICKSCORE_HPP_GUARD
#define PROBABILISTIC_R_MODELLING_TICKSCORE_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_ManufactureTickScore(SEXP probability);
}

#endif

