#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <boost/algorithm/string/replace.hpp>
#include <torch/torch.h>
#include <libtorch_support/Buffers.hpp>
#include <libtorch_support/Parameterisation.hpp>
#include <modelling/distribution/Normal.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/model/AutoRegressive.hpp>
#include <modelling/model/ARARCHTX.hpp>

#include <log/trivial.hpp>

torch::Tensor var_transformation(const torch::Tensor& x, double vtcrimp, double vtcatch) {
    if (vtcrimp > 0.0) {
        auto zero = torch::full({}, 0.0, torch::kDouble);
        auto x_div_crimp = x.div(vtcrimp);
        return x_div_crimp.logaddexp(zero).mul(vtcrimp) + vtcatch*(x_div_crimp.square() + 1.0).reciprocal();
    } else if (vtcrimp == 0.0) {
        return x.clamp_min(0.0);
    } else {
        throw std::logic_error("vtcrimp < 0.0");
    }
}

torch::Tensor var_transformation_inv(const torch::Tensor& vtx, double vtcrimp) {
    if (vtcrimp > 0.0) {
        auto x = vtx.new_full(vtx.sizes(), std::numeric_limits<double>::quiet_NaN());
        auto lt_20_crimp = x.lt(20.0*vtcrimp);
        auto gt_20_crimp = lt_20_crimp.logical_not();
        x.index_put_({lt_20_crimp}, vtx.index({lt_20_crimp}).div(vtcrimp).exp().sub(1.0).log().mul(vtcrimp));
        x.index_put_({gt_20_crimp}, vtx.index({gt_20_crimp}));
        return x;
    } else if (vtcrimp == 0.0) {
        // Only a true inverse if vtx >= 0.0, otherwise gives largest x that
        // gives smallest vtx.
        return vtx.clamp_min(0.0);
    } else {
        throw std::logic_error("vtcrimp < 0.0");
    }
}

template<class VarParameterisation>
class ARARCHTX : public ProbabilisticCloneable<ARARCHTX<VarParameterisation>> {
    public:
        static std::unique_ptr<ARARCHTX<VarParameterisation>> FromNamedShapelyParameters(
            NamedShapelyParameters& parameters,
            Buffers& buffers
        ) {
            return std::unique_ptr<ARARCHTX<VarParameterisation>>(new ARARCHTX<VarParameterisation>(parameters, buffers));
        }

        void shapely_reset(void) override {
            if (mu->enabled()) mu = this->template register_shapely_parameter<Linear>(mu->name(), mu->shapely_parameter_clone());
            if (mean_exogenous_coef->enabled()) mean_exogenous_coef = this->template register_shapely_parameter<Linear>(mean_exogenous_coef->name(), mean_exogenous_coef->shapely_parameter_clone());
            ar = this->register_module(std::make_shared<AutoRegressive<Linear>>(*ar));
            if (sigma2->enabled()) sigma2 = this->template register_shapely_parameter<VarParameterisation>(sigma2->name(), sigma2->shapely_parameter_clone());
            if (var_exogenous_coef->enabled()) var_exogenous_coef = this->template register_shapely_parameter<Linear>(var_exogenous_coef->name(), var_exogenous_coef->shapely_parameter_clone());
            arch = this->register_module(std::make_shared<AutoRegressive<VarParameterisation>>(*arch));
            var_transformation_crimp = this->register_buffer("arch_transformation_crimp", var_transformation_crimp.clone());
            var_transformation_catch = this->register_buffer("arch_transformation_catch", var_transformation_catch.clone());
            regressand_name = this->register_buffer("regressand_name", regressand_name.clone());
            if (exogenous_name.numel()) exogenous_name = this->register_buffer("exogenous_name", exogenous_name.clone());
        }

        std::unique_ptr<Distribution> forward(const torch::OrderedDict<std::string, torch::Tensor>& observations) override {
            if (regressand_name.numel() == 0) {
                throw std::logic_error("observations_name.numel() == 0");
            }

            std::string regressand_name_str = static_cast<char *>(regressand_name.data_ptr());
            auto regressand = observations[regressand_name_str];
            auto regressand_sizes = regressand.sizes();

            auto exo = [&]() {
                if (mean_exogenous_coef->enabled() || var_exogenous_coef->enabled()) {
                    auto exo_nested = observations[static_cast<char *>(exogenous_name.data_ptr())];
                    auto exo_nested_sizes = exo_nested.sizes();
                    if (exo_nested_sizes.size() < regressand_sizes.size() || exo_nested_sizes.size() > regressand_sizes.size()+1) {
                        std::ostringstream ss;
                        ss << "exo_nested_sizes.size() < regressand_sizes.size() || exo_nested_sizes.size() > regressand_sizes.size()+1. "
                              "exo_nested_sizes.size() = " << exo_nested_sizes.size() << ", regressand_sizes.size() = " << regressand_sizes.size() << ".";
                        throw std::logic_error(ss.str());
                    }
                    for (decltype(regressand_sizes.size()) i = 0; i != regressand_sizes.size(); ++i) {
                        if (exo_nested_sizes.at(i) != regressand_sizes.at(i)) {
                            std::ostringstream ss;
                            ss << "exo_nested_sizes[i] (" << exo_nested_sizes[i] << ") != regressand_sizes[i] (" << regressand_sizes[i] << ").";
                            throw std::logic_error(ss.str());
                        }
                    }
                    if (exo_nested_sizes.size() == regressand_sizes.size()) exo_nested = exo_nested.unsqueeze(exo_nested_sizes.size());
                    return exo_nested;
                } else {
                    return torch::Tensor();
                }
            }();

            auto regressand_means = mu->enabled() ? mu->get() : regressand.new_full({}, 0.0);

            if (mu->enabled() && regressand_means.numel() > 1) {
                for (auto i = regressand_means.ndimension(); i < regressand.ndimension(); ++i) {
                    regressand_means = regressand_means.unsqueeze(i);
                }
                regressand_means = regressand_means.expand_as(regressand);
            }
            if (mean_exogenous_coef->enabled()) {
                regressand_means = missing::handle_na(
                    [](const torch::Tensor& m, const torch::Tensor& ed, const torch::Tensor& ec) {
                        return m + torch::matmul(ed, ec);
                    },
                    regressand_means,
                    exo,
                    mean_exogenous_coef->get()
                );
            }
            if (ar->enabled()) {
                regressand_means = missing::handle_na(
                    [](const torch::Tensor& m, const torch::Tensor& oar) {
                        return m + oar;
                    },
                    regressand_means,
                    ar->forward(regressand)
                );
            }

            auto regressand_std_devs = sigma2->enabled() ? sigma2->get() : regressand.new_full({}, 0.0);
            
            if (sigma2->enabled() && regressand_std_devs.numel() > 1) {
                for (auto i = regressand_std_devs.ndimension(); i < regressand.ndimension(); ++i) {
                    regressand_std_devs = regressand_std_devs.unsqueeze(i);
                }
                regressand_std_devs = regressand_std_devs.expand_as(regressand);
            }
            if (var_exogenous_coef->enabled()) {
                regressand_std_devs = missing::handle_na(
                    [](const torch::Tensor& o, const torch::Tensor& ed, const torch::Tensor& ec) {
                        return o + torch::matmul(ed, ec);
                    },
                    regressand_std_devs,
                    exo,
                    var_exogenous_coef->get()
                );
            }

            if (arch->enabled()) {
                auto residuals2 = missing::handle_na(
                    [](const torch::Tensor& o, const torch::Tensor& m) {
                        return (o - m).square();
                    },
                    regressand,
                    regressand_means
                );

                regressand_std_devs = missing::handle_na(
                    [](const torch::Tensor& osd, const torch::Tensor& oarch) {
                        return osd + oarch;
                    },
                    regressand_std_devs,
                    arch->forward(residuals2)
                );
            }
            auto vtcrimp = var_transformation_crimp.item<double>();
            auto vtcatch = var_transformation_catch.item<double>();
            regressand_std_devs = missing::handle_na(
                [vtcrimp, vtcatch](const torch::Tensor& osd) {
                    return torch::sqrt(var_transformation(osd, vtcrimp, vtcatch));
                },
                regressand_std_devs
            );

            return ManufactureNormal(
                {{regressand_name_str, regressand_means}},
                {{regressand_name_str, regressand_std_devs}}
            );
        }

        torch::OrderedDict<std::string, torch::Tensor> draw_observations(
            int64_t sample_size,
            int64_t burn_in_size,
            double first_draw
        ) const override {
            if (mean_exogenous_coef->enabled() || var_exogenous_coef->enabled()) {
                throw std::runtime_error("ARARCHTX::draw_observations not implemented for ARARCHTX models with exogenous coefficients.");
            }

            auto total_size = burn_in_size + sample_size;

            auto zero = torch::full({1}, 0.0, torch::kDouble);

            torch::Tensor mu_get = mu->enabled() ? mu->get().detach() : zero;
            torch::Tensor ar_get = ar->enabled() ? ar->get().detach() : zero;
            torch::Tensor sigma2_get = sigma2->enabled() ? sigma2->get().detach() : zero;
            torch::Tensor arch_get = arch->enabled() ? arch->get().detach() : zero;

            auto ar_sizes = ar_get.sizes();
            auto arch_sizes = arch_get.sizes();

            {
                const char* msg = "ARARCHTX::draw_observations not implemented for ARARCHTX models with a multi-dimensional index.";

                auto mu_sizes = mu_get.sizes();
                if (mu_sizes.size() > 1 || (mu_sizes.size() == 1 && mu_sizes.front() > 1)) {
                    throw std::runtime_error(msg);
                }

                if (ar_sizes.size() > 1) {
                    throw std::runtime_error(msg);
                }

                auto sigma2_sizes = sigma2_get.sizes();
                if (sigma2_sizes.size() > 1 || (sigma2_sizes.size() == 1 && sigma2_sizes.front() > 1)) {
                    throw std::runtime_error(msg);
                }

                if (arch_sizes.size() > 1) {
                    throw std::runtime_error(msg);
                }
            }

            std::string regressand_name_str = static_cast<char *>(regressand_name.data_ptr());

            auto ar_ord = ar_sizes.front();
            auto arch_ord = arch_sizes.front();
            auto ord = ar_ord + arch_ord;

            if (total_size <= ord) {
                return {{regressand_name_str, torch::full({sample_size}, first_draw, torch::kDouble)}};
            }

            auto z = at::normal(
                torch::full({total_size}, 0.0, torch::kDouble),
                torch::full({total_size}, 1.0, torch::kDouble)
            );

            auto mean = torch::full({total_size}, std::numeric_limits<double>::quiet_NaN(), torch::kDouble);
            auto out = torch::full({total_size}, std::numeric_limits<double>::quiet_NaN(), torch::kDouble);

            auto mu_d = mu_get.item<double>();
            auto ar_a = ar_get.accessor<double,1>();
            auto sigma2_d = sigma2_get.item<double>();
            auto arch_a = arch_get.accessor<double,1>();
            auto z_a = z.accessor<double,1>();
            auto mean_a = mean.accessor<double,1>();
            auto out_a = out.accessor<double,1>();

            auto get_mean_i = [&](auto i) {
                // Works for i >= ar_ord.
                auto mean_i = mu_d;
                for (decltype(ar_ord) j = 0; j != ar_ord; ++j) {
                    mean_i += ar_a[j]*out_a[i-1-j];
                }
                return mean_i;
            };

            auto get_var_i = [&](auto i) {
                // Works for i >= ord (ord = ar_ord + arch_ord).
                auto var_i = sigma2_d;
                for (decltype(arch_ord) j = 0; j != arch_ord; ++j) {
                    auto err = out_a[i-1-j] - mean_a[i-1-j];
                    var_i += arch_a[j]*err*err;
                }
                return var_i;
            };

            for (decltype(ar_ord) i = 0; i != ar_ord; ++i) {
                mean_a[i] = first_draw;
                out_a[i] = first_draw;
            }

            for (decltype(ord) i = ar_ord; i != ord; ++i) {
                mean_a[i] = get_mean_i(i);
                out_a[i] = first_draw;
            }

            for (decltype(total_size) i = ord; i != total_size; ++i) {
                auto mean_i = get_mean_i(i);
                auto var_i = get_var_i(i);
                mean_a[i] = mean_i;
                out_a[i] = std::sqrt(var_i)*z_a[i] + mean_i;
            }

            return {{regressand_name_str, out.index({torch::indexing::Slice(burn_in_size, total_size)})}};
        }

        torch::OrderedDict<std::string, torch::Tensor> barrier(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            torch::Tensor scaling
        ) const override {
            std::string regressand_name_str = static_cast<char *>(regressand_name.data_ptr());
            auto regressand = observations[regressand_name_str];

            auto expand_as_regressand = [&regressand](torch::Tensor&& x) {
                for (auto i = x.ndimension(); i < regressand.ndimension(); ++i) {
                    x = x.unsqueeze(i);
                }
                return x.expand_as(regressand);
            };

            auto barrier_out = torch::full(regressand.sizes(), 0.0);
            if (mu->enabled()) barrier_out += scaling*expand_as_regressand(mu->barrier());
            if (mean_exogenous_coef->enabled()) barrier_out += scaling*expand_as_regressand(mean_exogenous_coef->barrier());
            if (ar->enabled()) barrier_out += ar->barrier(scaling);
            if (sigma2->enabled()) barrier_out += scaling*expand_as_regressand(sigma2->barrier());
            if (var_exogenous_coef->enabled()) barrier_out += scaling*expand_as_regressand(var_exogenous_coef->barrier());
            if (arch->enabled()) barrier_out += arch->barrier(scaling);

            return {{std::move(regressand_name_str), std::move(barrier_out)}};
        }

        torch::OrderedDict<std::string, torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>>> observations_by_parameter(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            bool recursive = true,
            bool include_fixed = false
        ) const override {
            std::string regressand_name_str = static_cast<char *>(regressand_name.data_ptr());
            auto regressand = observations[regressand_name_str];

            torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>> ar_observations_by_parameter;
            torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>> arch_observations_by_parameter;
            if (recursive) {
                if (ar->enabled()) {
                    ar_observations_by_parameter = ar->observations_by_parameter(regressand, recursive);
                }
                if (arch->enabled()) {
                    arch_observations_by_parameter = arch->observations_by_parameter(regressand, recursive);
                }
            }

            auto mu_prev = mu->enabled() ? mu->get_prev() : torch::Tensor();
            auto sigma2_prev = sigma2->enabled() ? sigma2->get_prev() : torch::Tensor();

            torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>> out_regressand;
            int64_t out_regressand_size = 0;
            if (mu->enabled()) {
                if (mu_prev.numel() == 1) {
                    out_regressand_size += 1;
                } else {
                    out_regressand_size += mu_prev.sizes().at(0);
                }
            }
            out_regressand_size += mean_exogenous_coef->enabled();
            out_regressand_size += ar_observations_by_parameter.size();
            if (sigma2->enabled()) {
                if (sigma2_prev.numel() == 1) {
                    out_regressand_size += 1;
                } else {
                    out_regressand_size += sigma2_prev.sizes().at(0);
                }
            }
            out_regressand_size += var_exogenous_coef->enabled();
            out_regressand_size += arch_observations_by_parameter.size();
            out_regressand.reserve(out_regressand_size);

            auto append_regressand_by_param_mu_sigma2 = [&](const auto& x, const auto& x_prev) {
                if (x->enabled()) {
                    if (x_prev.numel() == 1) {
                        out_regressand.insert(shapely_parameter_raw_name(*x), {std::vector<torch::indexing::TensorIndex>(regressand.ndimension(), torch::indexing::Slice())});
                    } else {
                        if (x_prev.ndimension() != 1) {
                            throw std::logic_error("x_prev.ndimension() != 1");
                        }

                        auto x_size = x_prev.sizes().at(0);

                        std::vector<std::vector<torch::indexing::TensorIndex>> out_x; out_x.reserve(x_size);
                        auto this_location = torch::full({x_size}, false, torch::kBool);
                        auto this_location_a = this_location.template accessor<bool,1>();
                        for (int64_t i = 0; i != x_size; ++i) {
                            this_location_a[i] = true;

                            std::vector<torch::indexing::TensorIndex> out_x_i; out_x_i.reserve(regressand.ndimension());
                            out_x_i.emplace_back(this_location.clone());
                            for (int64_t j = 1; j < regressand.ndimension(); ++j) {
                                out_x_i.emplace_back(torch::indexing::Slice());
                            }
                            out_x.emplace_back(std::move(out_x_i));

                            this_location_a[i] = false;
                        }

                        out_regressand.insert(shapely_parameter_raw_name(*x), std::move(out_x));
                    }
                }
            };

            append_regressand_by_param_mu_sigma2(mu, mu_prev);

            if (mean_exogenous_coef->enabled()) {
                out_regressand.insert(shapely_parameter_raw_name(*mean_exogenous_coef), {std::vector<torch::indexing::TensorIndex>(regressand.ndimension(), torch::indexing::Slice())});
            }

            if (recursive && ar->enabled()) {
                this->observations_by_parameter_recursive_update(out_regressand, std::move(ar_observations_by_parameter), ar->name());
            }

            append_regressand_by_param_mu_sigma2(sigma2, sigma2_prev);

            if (var_exogenous_coef->enabled()) {
                out_regressand.insert(shapely_parameter_raw_name(*var_exogenous_coef), {std::vector<torch::indexing::TensorIndex>(regressand.ndimension(), torch::indexing::Slice())});
            }

            if (recursive && arch->enabled()) {
                this->observations_by_parameter_recursive_update(out_regressand, std::move(arch_observations_by_parameter), arch->name());
            }

            return {{std::move(regressand_name_str), std::move(out_regressand)}};
        }

    private:
        ARARCHTX(
            NamedShapelyParameters& shapely_parameters,
            Buffers& buffers
        ):
            torch::nn::Module([&]() {
                const auto& sp = shapely_parameters;
                auto& b = buffers;

                int64_t i = sp.idx;
                const auto& mu = sp.parameters[i++].value();
                const auto& mean_exogenous_coef = sp.parameters[i++].value();
                const auto& ar = sp.parameters[i++].value();
                const auto& sigma2 = sp.parameters[i++].value();
                const auto& var_exogenous_coef = sp.parameters[i++].value();
                const auto& arch = sp.parameters[i++].value();

                i = b.idx;
                auto var_transformation_crimp = b.buffers.at(i++).item<double>();
                auto var_transformation_catch = b.buffers.at(i++).item<double>();
                const auto *regressand_name = static_cast<const char*>(b.buffers.at(i++).data_ptr());

                auto ar_order = ar.enable ? ar.parameter.numel() : static_cast<decltype(ar.parameter.numel())>(0);
                auto arch_order = arch.enable ? arch.parameter.numel() : static_cast<decltype(arch.parameter.numel())>(0);

                auto order = [](const auto& p) {
                    return p.enable ? p.parameter.numel() : static_cast<decltype(p.parameter.numel())>(0);
                };

                std::ostringstream ss;
                ss << "ARARCHTX("
                   << regressand_name << ", "
                   << order(mu) << ", "
                   << order(mean_exogenous_coef) << ", "
                   << order(ar) << ", "
                   << order(sigma2) << ", "
                   << order(var_exogenous_coef) << ", "
                   << order(arch) << ", "
                   << var_transformation_crimp << ", "
                   << var_transformation_catch
                   << ")";
                auto ss_str = ss.str();
                boost::algorithm::replace_all(ss_str, ".", "<dot>"); // Submodule names cannot contain a '.' character.

                return(ss_str);
            }()),
            mu(this->template register_next_shapely_parameter<Linear>(shapely_parameters)),
            mean_exogenous_coef(this->template register_next_shapely_parameter<Linear>(shapely_parameters)),
            ar(this->register_module(std::make_shared<AutoRegressive<Linear>>(shapely_parameters, "ar"))),
            sigma2(this->template register_next_shapely_parameter<VarParameterisation>(shapely_parameters)),
            var_exogenous_coef(this->template register_next_shapely_parameter<Linear>(shapely_parameters)),
            arch(this->register_module(std::make_shared<AutoRegressive<VarParameterisation>>(shapely_parameters, "arch"))),
            var_transformation_crimp(this->register_next_buffer("arch_transformation_crimp", buffers)),
            var_transformation_catch(this->register_next_buffer("arch_transformation_catch", buffers)),
            regressand_name(this->register_next_buffer("regressand_name", buffers)),
            exogenous_name((mean_exogenous_coef->enabled() || var_exogenous_coef->enabled()) ? this->register_next_buffer("exogenous_name", buffers) : torch::Tensor())
        { }

        std::shared_ptr<Linear> mu;
        std::shared_ptr<Linear> mean_exogenous_coef;
        std::shared_ptr<AutoRegressive<Linear>> ar;
        std::shared_ptr<VarParameterisation> sigma2;
        std::shared_ptr<Linear> var_exogenous_coef;
        std::shared_ptr<AutoRegressive<VarParameterisation>> arch;

        torch::Tensor var_transformation_crimp;
        torch::Tensor var_transformation_catch;

        torch::Tensor regressand_name;
        torch::Tensor exogenous_name;
};

std::unique_ptr<ProbabilisticModule> ManufactureARARCHTX(
    NamedShapelyParameters& sp,
    Buffers& b
) {
    const auto& var_transformation_crimp = b.buffers.at(b.idx);
    const auto& arch = sp.parameters[sp.idx + 5].value();
    if (var_transformation_crimp.item<double>() == 0.0) {
        return ARARCHTX<SigmoidUnboundedAbove>::FromNamedShapelyParameters(sp, b);
    }
    return ARARCHTX<Linear>::FromNamedShapelyParameters(sp, b);
}

std::unique_ptr<ProbabilisticModule> ManufactureARARCHTX(
    NamedShapelyParameters& sp,
    Buffers& b,
    const torch::OrderedDict<std::string, torch::Tensor>& observations
) {
    // Initialise parameters in sp by their least-squares parameter estimates.

    int64_t i = b.idx;
    auto& var_transformation_crimp = b.buffers.at(i++);
    auto& var_transformation_catch = b.buffers.at(i++);
    auto& regressand_name_tensor = b.buffers.at(i++);
    auto exo = [&]() {
        if (b.buffers.size() > i-b.idx) {
            auto& exo_name_tensor = b.buffers.at(i++);
            const char *exo_name = static_cast<char *>(exo_name_tensor.data_ptr());
            return observations[exo_name];
        } else {
            return torch::Tensor();
        }
    }();

    i = sp.idx;
    auto& mu = sp.parameters[i++].value();
    auto& mean_exogenous_coef = sp.parameters[i++].value();
    auto& ar = sp.parameters[i++].value();
    auto& sigma2 = sp.parameters[i++].value();
    auto& var_exogenous_coef = sp.parameters[i++].value();
    auto& arch = sp.parameters[i++].value();

    std::string regressand_name_str = static_cast<char *>(regressand_name_tensor.data_ptr());
    auto regressand = observations[regressand_name_str];
    auto regressand_numel = regressand.numel();
    auto regressand_sizes = regressand.sizes();
    auto regressand_sizes_0 = regressand_sizes.at(0);
    auto t_size = regressand_sizes.back();

    torch::Tensor X_re = torch::empty({regressand_numel, 0}, torch::kDouble);
    if ((mu.enable && mu.parameter.numel() > 1) || (sigma2.enable && sigma2.parameter.numel() > 1)) {
        auto regressand_sizes_0 = regressand_sizes.at(0);
        std::vector<int64_t> X_re_sizes; X_re_sizes.reserve(regressand.ndimension()+1);
        for (const auto& s : regressand_sizes) {
            X_re_sizes.emplace_back(s);
        }
        X_re_sizes.emplace_back(regressand_sizes_0);

        X_re = regressand.new_full(X_re_sizes, 0.0);
        for (decltype(regressand_sizes_0) i = 0; i != regressand_sizes_0; ++i) {
            X_re.index_put_({i, torch::indexing::Ellipsis, i}, 1.0);
        }

        X_re = X_re.flatten(0, regressand.ndimension()-1);
    }

    torch::Tensor X_mu = torch::empty({regressand_numel, 0}, torch::kDouble);
    if (mu.enable) {
        if (mu.parameter.numel() > 1) {
            X_mu = X_re;
        } else {
            X_mu = regressand.new_full({regressand_sizes}, 1.0).flatten().unsqueeze(1);
        }
    }

    torch::Tensor X_exo = torch::empty({regressand_numel, 0}, torch::kDouble);
    if (mean_exogenous_coef.enable) {
        X_exo = exo.flatten(0, regressand.ndimension()-1);
    }

    torch::Tensor X_ar = torch::empty({regressand_numel, 0});
    if (ar.enable) {
        auto ar_order = ar.parameter.sizes().back();
        
        std::vector<int64_t> X_ar_sizes; X_ar_sizes.reserve(regressand.ndimension()+1);
        for (const auto& s : regressand_sizes) {
            X_ar_sizes.emplace_back(s);
        }
        X_ar_sizes.emplace_back(ar_order);

        X_ar = regressand.new_full(X_ar_sizes, missing::na);
        for (decltype(ar_order) i = 0; i != ar_order; ++i) {
            X_ar.index_put_(
                {torch::indexing::Ellipsis, torch::indexing::Slice(i+1, t_size), i},
                regressand.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, t_size - i - 1)})
            );
        }

        X_ar = X_ar.flatten(0, regressand.ndimension()-1);
    }

    auto X_mean = torch::cat({X_mu, X_exo, X_ar}, 1);

    auto regressand_flat = regressand.flatten();
    auto present_rows_mean = torch::logical_and(X_mean.ne(missing::na).all(1), regressand_flat.ne(missing::na));
    auto X_mean_present = X_mean.index({present_rows_mean, torch::indexing::Slice()});
    auto regressand_flat_present = regressand_flat.index({present_rows_mean});

    if (static_cast<torch::Tensor>(X_mean_present.eq(missing::na).any()).item<bool>() || static_cast<torch::Tensor>(regressand_flat_present.eq(missing::na).any()).item<bool>()) {
        throw std::logic_error("X_mean_present.eq(missing::na).any() || regressand_flat_present.eq(missing::na).any()");
    }

    torch::Tensor residuals_flat;
    if (X_mean.numel()) {
        auto params_mean = std::get<0>(regressand_flat_present.lstsq(X_mean_present)).squeeze().index({torch::indexing::Slice(0, X_mean_present.sizes().at(1))});
        int64_t i = 0;
        if (mu.enable) {
            mu.parameter = params_mean.index({torch::indexing::Slice(i, i + mu.parameter.numel())});
            i += mu.parameter.numel();
        }
        if (mean_exogenous_coef.enable) {
            mean_exogenous_coef.parameter = params_mean.index({torch::indexing::Slice(i, i + mean_exogenous_coef.parameter.numel())});
            i += mean_exogenous_coef.parameter.numel();
        }
        if (ar.enable) {
            ar.parameter = params_mean.index({torch::indexing::Slice(i, i + ar.parameter.numel())});
            i += ar.parameter.numel();
        }
        if (i != params_mean.numel()) {
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << X_mu.sizes() << " = X_mu.sizes()\n";
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << mu.parameter.numel() << " = mu.parameter.numel()\n";
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << mu.enable << " = mu.parameter.enable\n";

            PROBABILISTIC_LOG_TRIVIAL_DEBUG << X_exo.sizes() << " = X_exo.sizes()\n";
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << mean_exogenous_coef.parameter.numel() << " = mean_exogenous_coef.numel()\n";
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << mean_exogenous_coef.enable << " = mean_exogenous_parameter.enable\n";


            PROBABILISTIC_LOG_TRIVIAL_DEBUG << X_ar.sizes() << " = X_ar.sizes()\n";
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << ar.parameter.numel() << " = ar.parameter.numel()\n";
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << ar.enable << " = ar.enable\n";
            
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << X_mean.sizes() << " = X_mean.sizes()\n";

            std::ostringstream ss;
            ss << "i (" << i << ") != params_mean.numel() (" << params_mean.numel() << ')';
            throw std::logic_error(ss.str());
        }

        residuals_flat = missing::handle_na(
            [](const torch::Tensor& r, const torch::Tensor& x, const torch::Tensor& p) {
                return r - torch::matmul(x, p);
            },
            regressand_flat,
            X_mean,
            params_mean
        );
    } else {
        residuals_flat = regressand_flat;
    }

    auto residuals = residuals_flat.reshape(regressand_sizes);
    auto residuals_squared = residuals.square();
    auto residuals_squared_flat = residuals_flat.square();

    torch::Tensor X_sigma2 = torch::empty({regressand_numel, 0}, torch::kDouble);
    if (sigma2.enable) {
        if (sigma2.parameter.numel() > 1) {
            X_sigma2 = X_re;
        } else {
            X_sigma2 = regressand.new_full({regressand_sizes}, 1.0).flatten().unsqueeze(1);
        }
    }

    if (var_exogenous_coef.enable && !mean_exogenous_coef.enable) {
        X_exo = exo.flatten(0, regressand.ndimension()-1);
    }

    torch::Tensor X_arch = torch::empty({regressand_numel, 0}, torch::kDouble);
    if (arch.enable) {
        auto arch_order = arch.parameter.sizes().back();

        std::vector<int64_t> X_arch_sizes; X_arch_sizes.reserve(regressand.ndimension()+1);
        for (const auto& s : regressand_sizes) {
            X_arch_sizes.emplace_back(s);
        }
        X_arch_sizes.emplace_back(arch_order);

        X_arch = residuals.new_full(X_arch_sizes, missing::na);
        for (decltype(arch_order) i = 0; i != arch_order; ++i) {
            X_arch.index_put_(
                {torch::indexing::Ellipsis, torch::indexing::Slice(i+1, t_size), i},
                residuals_squared.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, t_size - i - 1)})
            );
        }

        X_arch = X_arch.flatten(0, regressand.ndimension()-1);
    }

    auto X_var = torch::cat({X_sigma2, X_exo, X_arch}, 1);
    auto present_rows_var = X_var.ne(missing::na).all(1).logical_and(residuals_squared_flat.ne(missing::na)).logical_and(residuals_squared_flat.gt(1e-30)); // add parameter to switch on or off the exp/ln.
    auto X_var_present = X_var.index({present_rows_var, torch::indexing::Slice()});
    auto residuals_squared_flat_present = residuals_squared_flat.index({present_rows_var});

    if (static_cast<torch::Tensor>(X_var_present.eq(missing::na).any()).item<bool>() || static_cast<torch::Tensor>(residuals_squared_flat_present.eq(missing::na).any()).item<bool>()) {
        throw std::logic_error("X_var_present.eq(missing::na).any() || residuals_squared_flat_present.eq(missing::na).any()");
    }

    auto vtcrimp = var_transformation_crimp.item<double>();

    /*
    auto transformed_residuals_squared_flat_present = residuals_squared_flat_present.new_full(residuals_squared_flat_present.sizes(), std::numeric_limits<double>::quiet_NaN());
    auto lt_20_crimp = residuals_squared_flat_present.lt(20.0*vtcrimp);
    auto gt_20_crimp = lt_20_crimp.logical_not();
    transformed_residuals_squared_flat_present.index_put_({lt_20_crimp}, residuals_squared_flat_present.index({lt_20_crimp}).div(vtcrimp).exp().sub(1.0).log().mul(vtcrimp));
    transformed_residuals_squared_flat_present.index_put_({gt_20_crimp}, residuals_squared_flat_present.index({gt_20_crimp}));
    */
    auto transformed_residuals_squared_flat_present = var_transformation_inv(residuals_squared_flat_present, vtcrimp);
    auto params_var = std::get<0>(transformed_residuals_squared_flat_present.lstsq(X_var_present)).squeeze().index({torch::indexing::Slice(0, X_var_present.sizes().at(1))});
    {
        int64_t i = 0;
        if (sigma2.enable) {
            sigma2.parameter = params_var.index({torch::indexing::Slice(i, i + sigma2.parameter.numel())});
            i += sigma2.parameter.numel();
        }
        if (var_exogenous_coef.enable) {
            var_exogenous_coef.parameter = params_var.index({torch::indexing::Slice(i, i + var_exogenous_coef.parameter.numel())});
            i += var_exogenous_coef.parameter.numel();
        }
        if (arch.enable) {
            arch.parameter = params_var.index({torch::indexing::Slice(i, i + arch.parameter.numel())});
            i += arch.parameter.numel();
        }
        if (i != params_var.numel()) {
            std::ostringstream ss;
            ss << "i (" << i << ") != params_var.numel() (" << params_var.numel() << ')';
            throw std::logic_error(ss.str());
        }
    }

    if (var_transformation_crimp.item<double>() == 0.0) {
        return ARARCHTX<SigmoidUnboundedAbove>::FromNamedShapelyParameters(sp, b);
    }
    return ARARCHTX<Linear>::FromNamedShapelyParameters(sp, b);
}

