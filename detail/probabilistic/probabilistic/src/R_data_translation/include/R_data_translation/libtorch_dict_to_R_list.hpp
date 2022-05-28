#ifndef PROBABILISTIC_R_DATA_TRANSLATION_LIBTORCH_DICT_TO_LIST_HPP_GUARD
#define PROBABILISTIC_R_DATA_TRANSLATION_LIBTORCH_DICT_TO_LIST_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_libtorch_dict_to_list(SEXP libtorch_dict_R);
}

#endif

