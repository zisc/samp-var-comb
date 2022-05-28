#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <R_data_translation/R_list_to_libtorch_dict.hpp>

bool is_missing_R(int x) {
    return x == NA_INTEGER;
}

bool is_missing_R(double x) {
    return ISNAN(x);
}

template<typename T>
T convert_missing_R(T x) {
    if (is_missing_R(x)) {
        return missing::na;
    } else {
        return x;
    }
}

auto R_to_libtorch_dict(
    SEXP R_data,
    SEXP R_index
) {
    if (!Rf_isNewList(R_data)) {
        throw std::logic_error("R_to_libtorch_dict: !Rf_isNewList(R_data)");
    }

    if (!Rf_isString(R_index)) {
        throw std::logic_error("R_to_libtorch_dict: !Rf_isString(R_index)");
    }

    auto R_data_length = Rf_length(R_data);
    auto R_index_length = Rf_length(R_index);
    auto R_data_names = Rf_getAttrib(R_data, R_NamesSymbol);

    if (R_data_length <= 0) {
        throw std::logic_error("R_to_libtorch_dict: Rf_length(R_data) <= 0");
    }

    if (R_index_length <= 0) {
        throw std::logic_error("R_to_libtorch_dict: Rf_length(R_index) <= 0");
    }

    auto R_data_nrows = Rf_length(VECTOR_ELT(R_data, 0));
    for (decltype(R_data_length) i = 1; i != R_data_length; ++i) {
        if (Rf_length(VECTOR_ELT(R_data, i)) != R_data_nrows) {
            throw std::logic_error("R_to_libtorch_dict: R_data has elements of different lengths.");
        }
    }

    auto index = [&]() {
        std::vector<decltype(R_index_length)> out;
        out.reserve(R_index_length);
        for (decltype(R_index_length) i = 0; i != R_index_length; ++i) {
            auto R_index_i = CHAR(STRING_ELT(R_index, i));
            decltype(R_data_length) j;
            for (j = 0; j != R_data_length; ++j) {
                auto R_data_names_j = CHAR(STRING_ELT(R_data_names, j));
                if (strcmp(R_data_names_j, R_index_i) == 0) {
                    out.emplace_back(j);
                    break;
                }
            }
            if (j == R_data_length) {
                std::ostringstream ss;
                ss << "R_to_libtorch_dict: index name \"" << R_index_i << "\" not found in data.";
                throw std::logic_error(ss.str());
            }
        }
        return out;
    }();

    for (const auto i : index) {
        if (!Rf_isInteger(VECTOR_ELT(R_data, i))) {
            throw std::logic_error("R_to_libtorch_dict: not all index columns are of integer type.");
        }
    }

    auto index_values = [&]() {
        std::vector<int*> out;
        out.reserve(R_index_length);
        for (const auto i : index) {
            out.emplace_back(INTEGER(VECTOR_ELT(R_data, i)));
        }
        return out;
    }();

    auto measurable = [&]() {
        std::vector<decltype(R_data_length)> out;
        out.reserve(R_data_length - R_index_length);
        for (decltype(R_data_length) i = 0; i != R_data_length; ++i) {
            if (std::find(index.cbegin(), index.cend(), i) == index.cend()) {
                out.emplace_back(i);
            }
        }
        return out;
    }();

    auto tensors_size = [&]() {
        std::vector<int64_t> out;
        out.reserve(index.size());
        for (const auto& v : index_values) {
            out.emplace_back(*std::max_element(v, v + R_data_nrows) + 1);
        }
        return out;
    }();

    auto row_major_index = [&](auto R_data_row, auto numel) {
        int64_t out = 0;
        int64_t stride = 1;
        for (int64_t i = tensors_size.size()-1; i >= 0; --i) {
            out += stride*index_values[i][R_data_row];
            stride *= tensors_size[i];
        }
        #ifndef NDEBUG
            if (out >= numel) {
                std::ostringstream ss;
                ss << "R_to_libtorch_dict: row_major_index (" << out << ") >= numel (" << numel << ") for "
                      "R_data_row = " << R_data_row << ", "
                      "index_values = [";
                for (int64_t i = 0; i < tensors_size.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << index_values[i][R_data_row];
                }
                ss << "], "
                      "tensors_size = [";
                for (int64_t i = 0; i < tensors_size.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << tensors_size[i];
                }
                ss << "].";
                throw std::logic_error(ss.str());
            }
        #endif
        return out;
    };

    auto measurable_to_tensor = [&](auto tensor_data_ptr, auto numel, auto measurable) {
        for (int64_t i = 0; i != R_data_nrows; ++i) {
            tensor_data_ptr[row_major_index(i, numel)] = convert_missing_R(measurable[i]);
        }
    };

    torch::OrderedDict<std::string, torch::Tensor> out;
    for (auto m : measurable) {
        auto R_data_m = VECTOR_ELT(R_data, m);
        auto name = CHAR(STRING_ELT(R_data_names, m));
        auto tensor = [&]() {
            /* if (Rf_isLogical(R_data_m)) {
                auto out = torch::full(tensors_size, missing::na, torch::kBool);
                measurable_to_tensor(out.data_ptr<bool>(), out.numel(), LOGICAL(R_data_m));
                return out;
            } else if (Rf_isInteger(R_data_m)) {
                auto out = torch::full(tensors_size, missing::na, torch::kLong);
                measurable_to_tensor(out.data_ptr<int64_t>(), out.numel(), INTEGER(R_data_m));
                return out;
            } else */ if (Rf_isReal(R_data_m)) {
                auto out = torch::full(tensors_size, missing::na, torch::kDouble);
                measurable_to_tensor(out.data_ptr<double>(), out.numel(), REAL(R_data_m));
                return out;
            } else {
                throw std::logic_error("R_to_libtorch_dict: unsupported type, only real measured variables supported.");
            }
        }();
        out.insert(name, tensor);
    }
    
    return out;
}

SEXP R_new_libtorch_dict(void) { return R_handle_exception([]() {
    R_protect_guard protect_guard;
    return shared_ptr_to_EXTPTRSXP(std::make_shared<torch::OrderedDict<std::string, torch::Tensor>>(), protect_guard);
});}

SEXP R_libtorch_dict_append_list(
    SEXP libtorch_dict_R,
    SEXP R_data,
    SEXP R_index
) { return R_handle_exception([&]() {
    auto libtorch_dict = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(libtorch_dict_R);
    auto new_tensors = R_to_libtorch_dict(R_data, R_index);
    libtorch_dict->update(new_tensors);

    return R_NilValue;
});}

SEXP R_libtorch_dict_append_dict(
    SEXP dict_lhs_R,
    SEXP dict_rhs_R
) { return R_handle_exception([&]() {
    auto dict_lhs = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(dict_lhs_R);
    auto dict_rhs = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(dict_rhs_R);
    dict_lhs->update(*dict_rhs);
    return R_NilValue;
});}

SEXP R_libtorch_dict_combine(
    SEXP dict_one_R,
    SEXP dict_two_R
) { return R_handle_exception([&]() {
    auto dict_one = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(dict_one_R);
    auto dict_two = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(dict_two_R);
    auto dict_out = std::make_unique<torch::OrderedDict<std::string, torch::Tensor>>();
    dict_out->update(*dict_one);
    dict_out->update(*dict_two);
    R_protect_guard protect_guard;
    return shared_ptr_to_EXTPTRSXP(std::move(dict_out), protect_guard);
});}

SEXP R_libtorch_dict_time_slice(
    SEXP libtorch_dict_R,
    SEXP t_begin_R,
    SEXP t_end_R
) { return R_handle_exception([&]() {
    auto libtorch_dict = EXTPTRSXP_to_shared_ptr<torch::OrderedDict<std::string, torch::Tensor>>(libtorch_dict_R);
    auto t_begin = INTEGER(t_begin_R)[0];
    auto t_end = INTEGER(t_end_R)[0];
    auto dict_out = std::make_unique<torch::OrderedDict<std::string, torch::Tensor>>();
    dict_out->reserve(libtorch_dict->size());
    for (const auto& item : *libtorch_dict) {
        dict_out->insert(item.key(), item.value().index({torch::indexing::Ellipsis, torch::indexing::Slice(t_begin, t_end)}));
    }
    R_protect_guard protect_guard;
    return shared_ptr_to_EXTPTRSXP(std::move(dict_out), protect_guard);
});}

