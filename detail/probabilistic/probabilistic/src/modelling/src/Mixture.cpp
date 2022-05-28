#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <R_protect_guard.hpp>
#include <R_support/function.hpp>
#include <data_translation/libtorch_tensor_to_R_list.hpp>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <libtorch_support/indexing.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/Mixture.hpp>

class Mixture : public Distribution {
    public:
        Mixture(
            std::vector<std::shared_ptr<Distribution>> components_in,
            torch::Tensor weights_in
        ):
            components(components_in),
            weights(weights_in)
        { }

        torch::OrderedDict<std::string, torch::Tensor> density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            return mix([&observations](const Distribution& d) { return d.density(observations); });
        }

        torch::OrderedDict<std::string, torch::Tensor> density(
            double observations
        ) const override {
            return mix([&observations](const Distribution& d) { return d.density(observations); });
        }

        torch::OrderedDict<std::string, torch::Tensor> log_density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            return log_mix([&observations](const Distribution& d) { return d.log_density(observations); });
        }

        torch::OrderedDict<std::string, torch::Tensor> log_density(
            double observations
        ) const override {
            return log_mix([&observations](const Distribution& d) { return d.log_density(observations); });
        }

        torch::OrderedDict<std::string, torch::Tensor> cdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            return mix([&observations](const Distribution& d) { return d.cdf(observations); });
        }

        torch::OrderedDict<std::string, torch::Tensor> log_cdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            return log_mix([&observations](const Distribution& d) { return d.log_cdf(observations); });
        }

        torch::OrderedDict<std::string, torch::Tensor> ccdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            return mix([&observations](const Distribution& d) { return d.ccdf(observations); });
        }

        torch::OrderedDict<std::string, torch::Tensor> log_ccdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            return log_mix([&observations](const Distribution& d) { return d.log_ccdf(observations); });
        }

        torch::OrderedDict<std::string, torch::Tensor> interval_probability(
            const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
            const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
        ) const override {
            return mix([&open_lower_bound, &closed_upper_bound](const Distribution& d) {
                return d.interval_probability(open_lower_bound, closed_upper_bound);
            });
        }

        torch::OrderedDict<std::string, torch::Tensor> interval_complement_probability(
            const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
            const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
        ) const override {
            return mix([&open_lower_bound, &closed_upper_bound](const Distribution& d) {
                return d.interval_complement_probability(open_lower_bound, closed_upper_bound);
            });
        }

        torch::OrderedDict<std::string, torch::Tensor> log_interval_probability(
            const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
            const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
        ) const override {
            return log_mix([&open_lower_bound, &closed_upper_bound](const Distribution& d) {
                return d.log_interval_probability(open_lower_bound, closed_upper_bound);
            });
        }

        torch::OrderedDict<std::string, torch::Tensor> log_interval_complement_probability(
            const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
            const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
        ) const override {
            return log_mix([&open_lower_bound, &closed_upper_bound](const Distribution& d) {
                return d.log_interval_complement_probability(open_lower_bound, closed_upper_bound);
            });
        }

        torch::OrderedDict<std::string, torch::Tensor> draw(void) const override {
            auto structure = get_structure();

            std::vector<torch::OrderedDict<std::string, torch::Tensor>> component_draws;
            component_draws.reserve(components.size());
            for (const auto& c : components) {
                component_draws.emplace_back(c->draw());
            }

            torch::OrderedDict<std::string, torch::Tensor> output; output.reserve(structure.size());
            for (const auto& item : structure) {
                auto series_i = item.key();
                const auto& structure_i = item.value();

                auto weights_expanded = weights.detach().clone();
                for (int64_t i = 0; i != structure_i.size(); ++i) {
                    weights_expanded.unsqueeze_(0);
                }
                auto expanded_size = structure_i;
                expanded_size.emplace_back(weights.sizes().front());
                weights_expanded = weights_expanded.expand(expanded_size);

                auto multinomial_draw = torch::multinomial(weights_expanded, 1, true).squeeze();

                auto output_i = torch::full(structure_i, std::numeric_limits<double>::quiet_NaN(), torch::kDouble);
                for (int64_t j = 0; j != components.size(); ++j) {
                    auto multinomial_draw_eq_j = multinomial_draw.eq(j);
                    output_i.index_put_({multinomial_draw_eq_j}, component_draws.at(j)[series_i].index({multinomial_draw_eq_j}));
                }

                output.insert(std::move(series_i), std::move(output_i));
            }

            return output;
        }

        torch::OrderedDict<std::string, std::vector<int64_t>> get_structure(void) const override {
            auto num_components = components.size();
            
            std::vector<torch::OrderedDict<std::string, std::vector<int64_t>>> component_structures;
            component_structures.reserve(num_components);
            for (const auto& c : components) {
                component_structures.emplace_back(c->get_structure());
            }

            auto shared_series = get_common_keys(component_structures);

            torch::OrderedDict<std::string, std::vector<int64_t>> mixture_structure; mixture_structure.reserve(shared_series.size());
            for (const auto& series : shared_series) {
                mixture_structure.insert(series, component_structures.front()[series]);
            }

            return mixture_structure;
        }

        SEXP to_R_list(R_protect_guard& protect_guard) const override {
            auto weights_numel = weights.numel();
            SEXP weights_R = protect_guard.protect(Rf_allocVector(REALSXP, weights_numel));
            double *weights_R_a = REAL(weights_R);
            auto weights_a = weights.accessor<double, 1>();
            for (decltype(weights_numel) i = 0; i != weights_numel; ++i) {
                weights_R_a[i] = weights_a[i];
            }

            std::vector<SEXP> args; args.reserve(components.size()+1);
            args.emplace_back(weights_R);
            for (const auto& c : components) {
                args.emplace_back(c->to_R_list(protect_guard));
            }

            return call_R_function("probabilistic:::cpp_Mixture_to_dist_mixture", args, protect_guard);
        }

    private:
        std::vector<std::shared_ptr<Distribution>> components;
        torch::Tensor weights;

        template<class T>
        torch::OrderedDict<std::string, torch::Tensor> mix(T&& op) const {
            auto num_components = components.size();

            std::vector<torch::OrderedDict<std::string, torch::Tensor>> component_op_values; component_op_values.reserve(num_components);
            for (const auto& c : components) {
                component_op_values.emplace_back(op(*c));
            }

            auto shared_series = get_common_keys(component_op_values);

            torch::OrderedDict<std::string, torch::Tensor> mixture_op_value; mixture_op_value.reserve(shared_series.size());
            {
                std::vector<torch::Tensor> tensors; tensors.reserve(num_components);
                for (auto series : shared_series) {
                    for (const auto& op_value : component_op_values) {
                        auto op_value_series = op_value[series];
                        auto ndimension = op_value_series.ndimension();
                        tensors.emplace_back(op_value_series.unsqueeze(ndimension).unsqueeze(ndimension+1));
                    }
                    auto tensors_concatinated = torch::cat(tensors, tensors.front().ndimension()-1);
                    auto mixture_op_value_series_unsqueezed = missing::handle_na(
                        [](const torch::Tensor& tc, const torch::Tensor& w) {
                            return torch::matmul(tc, w);
                        },
                        tensors_concatinated,
                        weights
                    );
                    mixture_op_value.insert(
                        std::move(series),
                        mixture_op_value_series_unsqueezed.squeeze(mixture_op_value_series_unsqueezed.ndimension()-1)
                    );
                    tensors.clear();
                }
            }

            return mixture_op_value;
        }

        template<class T>
        torch::OrderedDict<std::string, torch::Tensor> log_mix(T&& log_op) const {
            auto num_components = components.size();
            auto log_weights = missing::handle_na([](const auto& w) { return w.log(); }, weights);

            std::vector<torch::OrderedDict<std::string, torch::Tensor>> component_log_op_values; component_log_op_values.reserve(num_components);
            for (const auto& c : components) {
                component_log_op_values.emplace_back(log_op(*c));
            }

            auto shared_series = get_common_keys(component_log_op_values);

            torch::OrderedDict<std::string, torch::Tensor> log_mixture_op_value; log_mixture_op_value.reserve(shared_series.size());
            {
                std::vector<torch::Tensor> tensors; tensors.reserve(num_components);
                for (auto series : shared_series) {
                    for (const auto& log_op_value : component_log_op_values) {
                        auto log_op_value_series = log_op_value[series];
                        auto ndimension = log_op_value_series.ndimension();
                        tensors.emplace_back(log_op_value_series.unsqueeze(log_op_value_series.ndimension()));
                    }
                    auto tensors_concatinated = torch::cat(tensors, tensors.front().ndimension()-1);
                    log_mixture_op_value.insert(
                        std::move(series),
                        missing::handle_na(
                            [](const torch::Tensor& tc, const torch::Tensor& lw) {
                                return (lw + tc).logsumexp({tc.ndimension()-1});
                            },
                            tensors_concatinated,
                            log_weights
                        )
                    );
                    tensors.clear();
                }
            }

            return log_mixture_op_value;
        }
};

std::unique_ptr<Distribution> ManufactureMixture(
    std::vector<std::shared_ptr<Distribution>> components,
    torch::Tensor weights
) {
    if (weights.sizes().size() != 1) {
        throw std::logic_error("weights.sizes().size() != 1");
    }

    if (weights.numel() != components.size()) {
        throw std::logic_error("weights.numel() != components.size()");
    }

    if (static_cast<torch::Tensor>(weights.le(0.0).any()).item<bool>()) {
        throw std::logic_error("weights.leq(0.0).any().item<bool>()");
    }

    weights = weights/weights.sum();

    return std::make_unique<Mixture>(std::move(components), std::move(weights));
}

