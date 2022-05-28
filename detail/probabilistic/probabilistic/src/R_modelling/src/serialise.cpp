#include <sstream>
#include <stdexcept>
#include <R.h>
#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <log/trivial.hpp>
#include <torch/torch.h>
#include <R_modelling/model/serialise.hpp>

SEXP R_serialise_model(
    SEXP models_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;

    auto nmodels = Rf_length(models_R);

    SEXP serialised_models_R = protect_guard.protect(Rf_allocVector(VECSXP, nmodels));

    for (decltype(nmodels) i = 0; i != nmodels; ++i) {
        SEXP models_R_i = VECTOR_ELT(models_R, i);
        auto model = EXTPTRSXP_to_shared_ptr<torch::nn::Module>(models_R_i);

        torch::serialize::OutputArchive archive;
        model->save(archive);

        std::ostringstream ss;
        archive.save_to(ss);

        auto ss_str = ss.str();
        auto ss_str_size = ss_str.size();

        SEXP model_serialised_R = protect_guard.protect(Rf_allocVector(RAWSXP, ss_str_size));
        std::memcpy(RAW(model_serialised_R), ss_str.data(), ss_str_size);

        SET_VECTOR_ELT(serialised_models_R, i, model_serialised_R);
    }

    return serialised_models_R;
});}

SEXP R_deserialise_model(
    SEXP models_out_R,
    SEXP models_serialised_R
) { return R_handle_exception([&](){
    R_protect_guard protect_guard;

    auto nmodels = Rf_length(models_out_R);
    if (nmodels != Rf_length(models_serialised_R)) {
        std::ostringstream ss;
        ss << "R_deserialise_model: Rf_length(models_out_R) != Rf_length(models_serialised_R) ("
           << nmodels << " != " << Rf_length(models_serialised_R) << ")";
        throw std::logic_error(ss.str());
    }

    for (decltype(nmodels) i = 0; i != nmodels; ++i) {
        SEXP models_out_R_i = VECTOR_ELT(models_out_R, i);
        SEXP models_serialised_R_i = VECTOR_ELT(models_serialised_R, i);

        torch::serialize::InputArchive archive;
        archive.load_from(reinterpret_cast<const char *>(RAW(models_serialised_R_i)), Rf_length(models_serialised_R_i));

        auto model = EXTPTRSXP_to_shared_ptr<torch::nn::Module>(models_out_R_i);
        model->load(archive);
    }

    return R_NilValue;
});}

