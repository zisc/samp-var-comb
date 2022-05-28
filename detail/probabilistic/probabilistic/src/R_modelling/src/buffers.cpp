#include <cstring>
#include <stdexcept>
#include <utility>
#include <torch/torch.h>
#include <Rinternals.h>
#include <libtorch_support/Buffers.hpp>
#include <R_modelling/buffers.hpp>

Buffers to_buffers(SEXP buffers_R) {
    if (!Rf_isNewList(buffers_R)) {
        throw std::logic_error("to_buffers: !Rf_isNewList(buffers_R)");
    }

    auto fill_tensor = [](auto tensor_data_ptr, auto numel, auto data) {
        for (decltype(numel) i = 0; i != numel; ++i) {
            tensor_data_ptr[i] = data[i];
        }
    };

    auto buffers_R_length = Rf_length(buffers_R);

    Buffers buffers;
    buffers.buffers.reserve(buffers_R_length);

    for(decltype(buffers_R_length) i = 0; i != buffers_R_length; ++i) {
        auto buffers_R_i = VECTOR_ELT(buffers_R, i);
        auto buffers_R_i_length = Rf_length(buffers_R_i);
        auto tensor = [&]() {
            if (Rf_isReal(buffers_R_i)) {
                auto out = torch::empty({buffers_R_i_length}, torch::kDouble);
                fill_tensor(out.data_ptr<double>(), out.numel(), REAL(buffers_R_i));
                return out;
            } else if (Rf_isInteger(buffers_R_i)) {
                auto out = torch::empty({buffers_R_i_length}, torch::kLong);
                fill_tensor(out.data_ptr<int64_t>(), out.numel(), INTEGER(buffers_R_i));
                return out;
            } else if (Rf_isLogical(buffers_R_i)) {
                auto out = torch::empty({buffers_R_i_length}, torch::kBool);
                fill_tensor(out.data_ptr<bool>(), out.numel(), LOGICAL(buffers_R_i));
                return out;
            } else if (Rf_isString(buffers_R_i)) {
                if (buffers_R_i_length != 1) {
                    throw std::logic_error("to_buffers: a buffer cannot contain more than one string.");
                }
                auto buffers_R_i_str = CHAR(STRING_ELT(buffers_R_i, 0));
                int64_t buffers_R_i_str_length = strlen(buffers_R_i_str) + 1; // "+ 1" for null-terminating character.
                auto out = torch::empty({buffers_R_i_str_length}, torch::kChar);
                fill_tensor(static_cast<char*>(out.data_ptr()), out.numel(), buffers_R_i_str); // unable to link to torch::Tensor::data_ptr<char>().
                                                                                               // check this again when upgrading the libtorch version.
                return out;
            } else {
                throw std::logic_error("Unrecognised R type in to_buffers function.");
            }
        }();
        buffers.buffers.emplace_back(std::move(tensor));
    }

    return buffers;
}

