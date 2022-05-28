#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <R_support/function.hpp>
#include <torch/torch.h>
#include <libtorch_support/logsubexp.hpp>
#include <libtorch_support/missing.hpp>
#include <data_translation/libtorch_tensor_to_R_list.hpp>
#include <modelling/distribution/Distribution.hpp>

#include <log/trivial.hpp>

constexpr double log_half = -0.693147180559945309417232121458176568075500134360255254120680009;

#ifndef NDEBUG
    class DestroyToFalse {
        public:
            DestroyToFalse(bool& flag_in): flag(flag_in) { }

            ~DestroyToFalse() {
                flag = false;
            }
        private:
            bool& flag;
    };
#endif

torch::OrderedDict<std::string, torch::Tensor> Distribution::full_as_observations(
    double x,
    const torch::OrderedDict<std::string, std::vector<int64_t>>& structure
) const {
    torch::OrderedDict<std::string, torch::Tensor> out; out.reserve(structure.size());
    for (const auto& item : structure) {
        out.insert(item.key(), torch::full(item.value(), x, torch::kDouble));
    }
    return out;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::full_as_observations(double x) const {
    return full_as_observations(x, get_structure());
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::density(
    const torch::OrderedDict<std::string, torch::Tensor>& observations
) const {
    #ifndef NDEBUG
        static bool parent_call = false;

        if (parent_call) {
            throw std::runtime_error("Distribution::density and Distribution::log_density undefined.");
        }

        DestroyToFalse falsifier(parent_call = true);
    #endif

    auto ret = log_density(observations);

    for (auto& item : ret) {
        auto& tensor = item.value();
        tensor = missing::handle_na(
            [](const torch::Tensor& x) { return x.exp(); },
            tensor
        );
    }
    
    return ret;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::density(
    double observations
) const {
    return density(full_as_observations(observations));
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_density(
    const torch::OrderedDict<std::string, torch::Tensor>& observations
) const {
    #ifndef NDEBUG
        static bool parent_call = false;

        if (parent_call) {
            throw std::runtime_error("Distribution::density and Distribution::log_density undefined.");
        }

        DestroyToFalse falsifier(parent_call = true);
    #endif

    auto ret = density(observations);

    for (auto& item : ret) {
        auto& tensor = item.value();
        tensor = missing::handle_na(
            [](const torch::Tensor& x) { return x.log(); },
            tensor
        );
    }
 
    return ret;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_density(
    double observations
) const {
    return log_density(full_as_observations(observations));
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::cdf(
    const torch::OrderedDict<std::string, torch::Tensor>& observations
) const {
    #ifndef NDEBUG
        static bool parent_call = false;

        if (parent_call) {
            throw std::runtime_error("Distribution::cdf and Distribution::log_cdf undefined.");
        }

        DestroyToFalse falsifier(parent_call = true);
    #endif

    auto ret = log_cdf(observations);

    for (auto& item : ret) {
        auto& tensor = item.value();
        tensor = missing::handle_na(
            [](const torch::Tensor& x) { return x.exp(); },
            tensor
        );
    }

    return ret;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::cdf(
    double observations
) const {
    return cdf(full_as_observations(observations));
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_cdf(
    const torch::OrderedDict<std::string, torch::Tensor>& observations
) const {
    #ifndef NDEBUG
        static bool parent_call = false;

        if (parent_call) {
            throw std::runtime_error("Distribution::cdf and Distribution::log_cdf undefined.");
        }

        DestroyToFalse falsifier(parent_call = true);
    #endif

    auto ret = cdf(observations);

    for (auto& item : ret) {
        auto& tensor = item.value();
        tensor = missing::handle_na(
            [](const torch::Tensor& x) { return x.log(); },
            tensor
        );
    }

    return ret;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_cdf(
    double observations
) const {
    return log_cdf(full_as_observations(observations));
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::ccdf(
    const torch::OrderedDict<std::string, torch::Tensor>& observations
) const {
    auto out = cdf(observations);
    for (auto& item : out) {
        auto& tensor = item.value();
        tensor = 1.0 - tensor;
    }
    return out;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::ccdf(
    double observations
) const {
    return ccdf(full_as_observations(observations));
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_ccdf(
    const torch::OrderedDict<std::string, torch::Tensor>& observations
) const {
    auto out = ccdf(observations);
    for (auto& item : out) {
        auto& out_i = item.value();
        out_i = out_i.log();
    }
    return out;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_ccdf(
    double observations
) const {
    return log_ccdf(full_as_observations(observations));
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::quantile(
    const torch::OrderedDict<std::string, torch::Tensor>& probabilities
) const {
    throw std::runtime_error("Distribution::quantile undefined.");
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::quantile(
    double probability
) const {
    return quantile(full_as_observations(probability));
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::interval_probability(
    const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
    const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
) const {
    auto cdf_lb = cdf(open_lower_bound);
    auto cdf_ub = cdf(closed_upper_bound);
    auto ccdf_lb = ccdf(open_lower_bound);
    auto ccdf_ub = ccdf(closed_upper_bound);

    torch::OrderedDict<std::string, torch::Tensor> out; out.reserve(cdf_lb.size());
    for (const auto& item : cdf_lb) {
        const auto& name = item.key();
        const auto& cdf_lb_i = item.value();
        const auto& cdf_ub_i = cdf_ub[name];
        const auto& ccdf_lb_i = ccdf_lb[name];
        const auto& ccdf_ub_i = ccdf_ub[name];
        auto sizes = cdf_lb_i.sizes();
        out.insert(
            name,
            missing::handle_na(
                [&sizes](
                    const auto& cdflbi,
                    const auto& cdfubi,
                    const auto& ccdflbi,
                    const auto& ccdfubi
                ) {
                    auto cdfsum = cdflbi + cdfubi;
                    return torch::where(
                        cdfubi.le(cdflbi),
                        torch::full(cdflbi.sizes(), 0.0, torch::kDouble),
                        torch::where(
                            (cdflbi+cdfubi).le(1.0),
                            cdfubi - cdflbi,
                            ccdflbi - ccdfubi
                        )
                    );
                },
                cdf_lb_i,
                cdf_ub_i,
                ccdf_lb_i,
                ccdf_ub_i
            )
        );
    }

    return out;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::interval_probability(
    double open_lower_bound,
    double closed_upper_bound
) const {
    auto structure = get_structure();

    if (open_lower_bound >= closed_upper_bound) {
        return full_as_observations(0.0, structure);
    }

    if (open_lower_bound == -std::numeric_limits<decltype(open_lower_bound)>::infinity()) {
        if (closed_upper_bound == std::numeric_limits<decltype(closed_upper_bound)>::infinity()) {
            return full_as_observations(1.0, structure);
        }
        return cdf(closed_upper_bound);
    }

    if (closed_upper_bound == std::numeric_limits<decltype(closed_upper_bound)>::infinity()) {
        return ccdf(open_lower_bound);
    }

    return interval_probability(
        full_as_observations(open_lower_bound, structure),
        full_as_observations(closed_upper_bound, structure)
    );   
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::interval_complement_probability(
    const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
    const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
) const {
    auto cdf_lb = cdf(open_lower_bound);
    auto ccdf_ub = ccdf(closed_upper_bound);

    torch::OrderedDict<std::string, torch::Tensor> out; out.reserve(cdf_lb.size());
    for (const auto& item : cdf_lb) {
        const auto& name = item.key();
        const auto& cdf_lb_i = item.value();
        const auto& ccdf_ub_i = ccdf_ub[name];
        out.insert(
            name,
            missing::handle_na(
                [](const auto& cdflbi, const auto& ccdfubi) {
                    return (cdflbi + ccdfubi).clamp_max(1.0);
                },
                cdf_lb_i,
                ccdf_ub_i
            )
        );
    }

    return out;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::interval_complement_probability(
    double open_lower_bound,
    double closed_upper_bound
) const {
    auto structure = get_structure();

    if (open_lower_bound >= closed_upper_bound) {
        return full_as_observations(1.0, structure);
    }

    if (open_lower_bound == -std::numeric_limits<decltype(open_lower_bound)>::infinity()) {
        if (closed_upper_bound == std::numeric_limits<decltype(closed_upper_bound)>::infinity()) {
            return full_as_observations(0.0, structure);
        }
        return ccdf(closed_upper_bound);
    }

    if (closed_upper_bound == std::numeric_limits<decltype(closed_upper_bound)>::infinity()) {
        return cdf(open_lower_bound);
    }

    return interval_complement_probability(
        full_as_observations(open_lower_bound, structure),
        full_as_observations(closed_upper_bound, structure)
    );
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_interval_probability(
    const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
    const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
) const {
    auto lcdf_lb = log_cdf(open_lower_bound);
    auto lcdf_ub = log_cdf(closed_upper_bound);
    auto lccdf_lb = log_ccdf(open_lower_bound);
    auto lccdf_ub = log_ccdf(closed_upper_bound);

    torch::OrderedDict<std::string, torch::Tensor> out; out.reserve(lcdf_lb.size());
    for (const auto& item : lcdf_lb) {
        const auto& name = item.key();
        const auto& lcdf_lb_i = item.value();
        const auto& lcdf_ub_i = lcdf_ub[name];
        const auto& lccdf_lb_i = lccdf_lb[name];
        const auto& lccdf_ub_i = lccdf_ub[name];
        auto laec = torch::logaddexp(lcdf_lb_i, lcdf_ub_i);
        
        auto laec_le_zero = laec.le(0.0);
        auto absent = missing::isna(lcdf_lb_i);
        auto present = absent.logical_not();
        auto use_c = torch::logical_and(present, laec_le_zero);
        auto use_cc = torch::logical_and(present, laec_le_zero.logical_not());

        auto ret = laec.new_empty(laec.sizes());
        ret.index_put_({use_c}, logsubexp(lcdf_ub_i.index({use_c}), lcdf_lb_i.index({use_c})));
        ret.index_put_({use_cc}, logsubexp(lccdf_lb_i.index({use_cc}), lccdf_ub_i.index({use_cc})));
        ret.index_put_({absent}, missing::na);
        out.insert(name, ret);
    }
    
    return out;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_interval_probability(
    double open_lower_bound,
    double closed_upper_bound
) const {
    auto structure = get_structure();

    if (open_lower_bound == -std::numeric_limits<decltype(open_lower_bound)>::infinity()) {
        if (closed_upper_bound == std::numeric_limits<decltype(closed_upper_bound)>::infinity()) {
            return full_as_observations(0.0, structure);
        }
        return log_cdf(closed_upper_bound);
    }

    if (closed_upper_bound == std::numeric_limits<decltype(closed_upper_bound)>::infinity()) {
        return log_ccdf(open_lower_bound);
    }

    return log_interval_probability(
        full_as_observations(open_lower_bound, structure),
        full_as_observations(closed_upper_bound, structure)
    );
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_interval_complement_probability(
    const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
    const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
) const {
    auto lcdf_lb = log_cdf(open_lower_bound);
    auto lccdf_ub = log_ccdf(closed_upper_bound);

    torch::OrderedDict<std::string, torch::Tensor> out; out.reserve(lcdf_lb.size());
    for (const auto& item : lcdf_lb) {
        const auto& name = item.key();
        const auto& lcdf_lb_i = lcdf_lb[name];
        const auto& lccdf_ub_i = lccdf_ub[name];
        out.insert(
            name,
            missing::handle_na(
                [] (const auto& lcdflbi, const auto& lccdfubi) {
                    return torch::logaddexp(lcdflbi, lccdfubi).clamp_max(0.0);
                },
                lcdf_lb_i,
                lccdf_ub_i
            )
        );
    }

    return out;
}

torch::OrderedDict<std::string, torch::Tensor> Distribution::log_interval_complement_probability(
    double open_lower_bound,
    double closed_upper_bound
) const {
    auto structure = get_structure();

    if (open_lower_bound >= closed_upper_bound) {
        return full_as_observations(0.0, structure);
    }

    if (open_lower_bound == -std::numeric_limits<decltype(open_lower_bound)>::infinity()) {
        return log_ccdf(closed_upper_bound);
    }

    if (closed_upper_bound == std::numeric_limits<decltype(closed_upper_bound)>::infinity()) {
        return log_cdf(open_lower_bound);
    }

    return log_interval_complement_probability(
        full_as_observations(open_lower_bound, structure),
        full_as_observations(closed_upper_bound, structure)
    );
}

SEXP to_R_list(
    const char *R_distributional_dist,
    const torch::Tensor& parameters,
    R_protect_guard& protect_guard
) {
    PersistentArgs args;

    args.tensor = missing::replace_na(parameters, NA_REAL);

    args.index_ndim = parameters.ndimension() - 1;
    // The indices 1, 2, ... along the final dimension of the parameters
    // tensor represent the 1st, 2nd, ... arguments to the R distribution
    // constructor, so subtract one to get the dimensionality of the index
    // to the distribution itself.

    if (args.index_ndim <= 0) {
        std::ostringstream ss;
        ss << "TensorsOfDistributions::to_R_list: args.distribution_index_ndim == "
           << args.index_ndim << " <= 0.";
        throw std::logic_error(ss.str());
    }

    args.dimension_sizes = parameters.sizes();

    args.tensor_index.resize(parameters.ndimension(), 0);

    int64_t R_dist_nargs = args.dimension_sizes.at(args.index_ndim);

    int64_t R_list_ncols = args.index_ndim + 1;
    // The first distribution_index_ndim columns store the index of the distribution,
    // in the parameters tensor, and the last column stores the R distribution itself.

    int64_t R_list_nrows = 1;
    for (int64_t i = 0; i != args.index_ndim ; ++i) {
        R_list_nrows *= parameters.sizes().at(i);
    }
    // The number of rows in the R_list is equal to the number of distributions
    // in the parameters tensor.

    SEXP ans = protect_guard.protect(Rf_allocVector(VECSXP, R_list_ncols));

    args.R_list_index_cols.reserve(args.index_ndim);
    for (int64_t i = 0; i != args.index_ndim; ++i) {
        SEXP col = Rf_allocVector(INTSXP, R_list_nrows);
        SET_VECTOR_ELT(ans, i, col);
        args.R_list_index_cols.emplace_back(INTEGER(col));
    }

    std::vector<SEXP> R_dist_args; R_dist_args.reserve(R_dist_nargs);
    args.R_list_data_cols.reserve(R_dist_nargs);
    for (int64_t i = 0; i != R_dist_nargs; ++i) {
        R_dist_args.emplace_back(protect_guard.protect(Rf_allocVector(REALSXP, R_list_nrows)));
        args.R_list_data_cols.emplace_back(REAL(R_dist_args.back()));
    }

    populate_R_list_rows(args);

    SEXP R_distributions = call_R_function(
        R_distributional_dist,
        R_dist_args,
        protect_guard
    );

    SET_VECTOR_ELT(ans, R_list_ncols-1, R_distributions);

    return ans;
}

SEXP Distribution::to_R_list(
    const char *R_distributional_dist,
    const torch::OrderedDict<std::string, torch::Tensor>& parameters,
    R_protect_guard& protect_guard
) const {
    auto nparams = parameters.size();
    SEXP ret = protect_guard.protect(Rf_allocVector(VECSXP, nparams));
    SEXP ret_names = Rf_allocVector(STRSXP, nparams);
    Rf_setAttrib(ret, R_NamesSymbol, ret_names);
    for (decltype(nparams) i = 0; i != nparams; ++i) {
        const auto& item = parameters[i];
        SET_VECTOR_ELT(ret, i, ::to_R_list(R_distributional_dist, item.value(), protect_guard));
        SET_STRING_ELT(ret_names, i, Rf_mkChar(item.key().c_str()));
    }
    return ret;
}

