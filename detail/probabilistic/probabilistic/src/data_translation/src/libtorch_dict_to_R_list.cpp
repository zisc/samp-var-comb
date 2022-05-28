#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <data_translation/libtorch_tensor_to_R_list.hpp>
#include <data_translation/libtorch_dict_to_R_list.hpp>

SEXP to_R_list(
    const torch::OrderedDict<std::string, torch::Tensor>& dict,
    R_protect_guard& protect_guard
) {
    SEXP ans = protect_guard.protect(Rf_allocVector(VECSXP, dict.size()));
    
    SEXP ans_names = Rf_allocVector(STRSXP, dict.size());
    Rf_setAttrib(ans, R_NamesSymbol, ans_names);

    for (int64_t i = 0; i != dict.size(); ++i) {
        const auto& item = dict[i];
        SET_VECTOR_ELT(ans, i, to_R_list(item.value(), protect_guard));
        SET_STRING_ELT(ans_names, i, Rf_mkChar(item.key().c_str()));
    }

    return ans;
}

