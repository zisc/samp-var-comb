#ifndef PROBABILISTIC_R_DATA_TRANSLATION_LIST_TO_LIBTORCH_DICT_HPP_GUARD
#define PROBABILISTIC_R_DATA_TRANSLATION_LIST_TO_LIBTORCH_DICT_HPP_GUARD

#include <Rinternals.h>
#include <dll_visibility.h>

extern "C" {
    DLL_PUBLIC SEXP R_new_libtorch_dict(void);

    DLL_PUBLIC SEXP R_libtorch_dict_append_list(
        SEXP libtorch_dict_R,
        SEXP R_data,
        SEXP R_index
    );

    DLL_PUBLIC SEXP R_libtorch_dict_append_dict(
        SEXP dict_lhs_R,
        SEXP dict_rhs_R
    );

    DLL_PUBLIC SEXP R_libtorch_dict_combine(
        SEXP dict_one_R,
        SEXP dict_two_R
    );

    DLL_PUBLIC SEXP R_libtorch_dict_time_slice(
        SEXP libtorch_dict_R,
        SEXP t_begin_R,
        SEXP t_end_R
    );
}

#endif

