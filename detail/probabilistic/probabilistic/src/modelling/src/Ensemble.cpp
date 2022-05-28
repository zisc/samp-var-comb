#include <memory>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <libtorch_support/Buffers.hpp>
#include <libtorch_support/Parameterisation.hpp>
#include <libtorch_support/indexing.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/Mixture.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/model/Ensemble.hpp>

template<class T>
std::string get_component_name(T i) {
    return std::string("components_") + std::to_string(i);
}

class Ensemble : public ProbabilisticCloneable<Ensemble> {
    public:
        static std::unique_ptr<Ensemble> FromNamedShapelyParameters(
            std::vector<std::shared_ptr<ProbabilisticModule>> components,
            NamedShapelyParameters& parameters,
            Buffers& buffers
        ) {
            return std::unique_ptr<Ensemble>(new Ensemble(std::move(components), parameters, buffers));
        }

        void shapely_reset(void) override {
            if (weights->enabled()) weights = register_shapely_parameter<decltype(weights)::element_type>(weights->name(), weights->shapely_parameter_clone());
            for (auto& item : components) { auto& c = item.value(); c = register_module(item.key(), c->clone_probabilistic_module()); }
            optimise_weights = register_buffer("optimise_weights", optimise_weights.clone());
            fixed_weights = register_buffer("fixed_weights", fixed_weights.clone());
            optimise_components = register_buffer("optimise_components", optimise_components.clone());
            fixed_components = register_buffer("fixed_components", fixed_components.clone());
        }

        std::unique_ptr<Distribution> forward(const torch::OrderedDict<std::string, torch::Tensor>& observations) override {
            std::vector<std::shared_ptr<Distribution>> component_distributions; component_distributions.reserve(components.size());
            for (auto& item : components) {
                component_distributions.emplace_back(item.value()->forward(observations));
            }
            return ManufactureMixture(component_distributions, weights->get());
        }

        torch::OrderedDict<std::string, torch::Tensor> barrier(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            torch::Tensor scaling
        ) const override {
            torch::OrderedDict<std::string, torch::Tensor> barrier_out; barrier_out.reserve(observations.size());
            for (const auto& item : components) {
                const auto& c = item.value();
                auto barrier_c = c->barrier(observations, scaling);
                for (auto& item : barrier_c) {
                    const auto& barrier_c_key = item.key();
                    auto *barrier_out_value = barrier_out.find(barrier_c_key);
                    if (barrier_out_value) {
                        *barrier_out_value += std::move(item.value());
                    } else {
                        barrier_out.insert(barrier_c_key, std::move(item.value()));
                    }
                }
            }
            auto weights_barrier = scaling*weights->barrier().mean();
            for (auto& item : barrier_out) {
                auto& v = item.value();
                v += weights_barrier;
            }
            return barrier_out;
        }

        bool fit(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            std::shared_ptr<const ScoringRule> scoring_rule,
            const FitPlan& plan,
            FitDiagnostics *diagnostics
        ) override {
            return ProbabilisticModule::fit(
                observations,
                std::move(scoring_rule),
                plan,
                get_parameters_to_optimise(),
                diagnostics
            );
        }

        torch::OrderedDict<std::string, torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>>> observations_by_parameter(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            bool recursive = true,
            bool include_fixed = false
        ) const override {
            auto fw = fixed_weights.item<bool>();
            auto fc = fixed_components.item<bool>();

            std::vector<decltype(components.front().value()->observations_by_parameter(observations,true))> components_obp;
            components_obp.reserve(components.size());
            for (const auto& item : components) {
                components_obp.emplace_back(item.value()->observations_by_parameter(observations, true));
            }

            auto shared_series = get_common_keys(components_obp);

            auto weights_raw_name = shapely_parameter_raw_name(weights->name());

            torch::OrderedDict<std::string, torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>>> out;
            out.reserve(shared_series.size());
            if (include_fixed || !fw) {
                out.reserve(shared_series.size());
                for (const auto& series : shared_series) {
                    const auto& obs = observations[series];
                    out.insert(series, {{weights_raw_name, {std::vector<torch::indexing::TensorIndex>(obs.ndimension(), torch::indexing::Slice())}}});
                }
            }

            if (recursive && (include_fixed || !fc)) {
                for (decltype(components.size()) i = 0; i != components.size(); ++i) {
                    const auto& component_name = components[i].key();
                    auto& cobp_i = components_obp.at(i);

                    decltype(cobp_i.size()) j = 0;
                    while (j != cobp_i.size()) {
                        auto series = cobp_i[j].key();
                        if (std::find(shared_series.cbegin(), shared_series.cend(), series) == shared_series.cend()) {
                            cobp_i.erase(series);
                        } else {
                            ++j;
                        }
                    }

                    observations_by_parameter_recursive_update(
                        out,
                        std::move(cobp_i),
                        component_name
                    );
                }
            }

            return out;
        }

        torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> estimating_equations_values(
            bool create_graph = false,
            bool recurse = true
        ) override {
            auto ow = optimise_weights.item<bool>();
            auto fw = fixed_weights.item<bool>();
            auto oc = optimise_components.item<bool>();
            auto fc = fixed_components.item<bool>();

            if (oc) {
                return ProbabilisticModule::estimating_equations_values(create_graph, recurse);
            } else {
                torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> out;

                if (!fw) {
                    out = ProbabilisticModule::estimating_equations_values(create_graph, false);
                }

                if (recurse && !fc) {
                    for (const auto& item : components) {
                        estimating_equations_values_recursive_update(
                            out,
                            item.value()->estimating_equations_values(create_graph, recurse),
                            item.key()
                        );
                    }
                }

                return out;
            }
        }

        std::vector<torch::Tensor> parameters(bool recurse = true, bool include_fixed = true) const override {
            if (include_fixed) {
                return torch::nn::Module::parameters(recurse);
            }

            auto fw = fixed_weights.item<bool>();
            auto fc = fixed_components.item<bool>();

            std::vector<torch::Tensor> out;
            if (!fw) {
                out = torch::nn::Module::parameters(false); // After this line, out contains weights.
            }

            std::vector<std::vector<torch::Tensor>> components_parameters;
            if (!fc) {
                components_parameters.reserve(components.size());
                for (const auto& item : components) {
                    components_parameters.emplace_back(item.value()->parameters());
                }
            }

            auto out_size = out.size();
            for (const auto& ps : components_parameters) {
                out_size += ps.size();
            }
            out.reserve(out_size);

            for (auto& ps : components_parameters) {
                for (auto& p : ps) {
                    out.emplace_back(std::move(p));
                }
            }

            return out;
        }

        torch::OrderedDict<std::string, torch::Tensor> named_parameters(bool recurse = true, bool include_fixed = true) const override {
            if (include_fixed) {
                return torch::nn::Module::named_parameters(recurse);
            }

            auto fw = fixed_weights.item<bool>();
            auto fc = fixed_components.item<bool>();

            torch::OrderedDict<std::string, torch::Tensor> out;
            if (!fw) {
                out = torch::nn::Module::named_parameters(false); // After this line, out contains weights;
            }

            std::vector<torch::OrderedDict<std::string, torch::Tensor>> component_named_parameters;
            if (!fc) {
                component_named_parameters.reserve(components.size());
                for (const auto& item : components) {
                    const auto& component_name = item.key();
                    auto prefix = component_name + ".";
                    auto without_prefix = item.value()->named_parameters(recurse);
                    decltype(without_prefix) with_prefix; with_prefix.reserve(without_prefix.size());
                    for (auto& p : without_prefix) {
                        with_prefix.insert(prefix + p.key(), std::move(p.value()));
                    }
                    component_named_parameters.emplace_back(std::move(with_prefix));
                }
            }

            auto out_size = out.size();
            for (const auto& nps : component_named_parameters) {
                out_size += nps.size();
            }
            out.reserve(out_size);

            for (auto& nps : component_named_parameters) {
                for (auto& item : nps) {
                    out.insert(item.key(), std::move(item.value()));
                }
            }

            return out;
        }

    private:
        Ensemble(
            std::vector<std::shared_ptr<ProbabilisticModule>> components_in,
            NamedShapelyParameters& shapely_parameters,
            Buffers& buffers
        ):
            weights(register_next_shapely_parameter<decltype(weights)::element_type>(shapely_parameters)),
            components([&]() {
                    torch::OrderedDict<std::string, std::shared_ptr<ProbabilisticModule>> components_out; components_out.reserve(components_in.size());
                for (decltype(components_in.size()) i = 0; i != components_in.size(); ++i) {
                    auto component_name = get_component_name(i);
                    components_out.insert(component_name, register_module(component_name, components_in[i]->clone_probabilistic_module()));
                }
                return components_out;
            }()),
            optimise_weights(register_next_buffer("optimise_weights", buffers)),
            fixed_weights(register_next_buffer("fixed_weights", buffers)),
            optimise_components(register_next_buffer("optimise_components", buffers)),
            fixed_components(register_next_buffer("fixed_components", buffers))
        {
            if (!optimise_weights.item<bool>() && !fixed_weights.item<bool>()) {
                throw std::logic_error("If weights is not to be optimised, then it must be treated as fixed.");
            }
        }

        torch::OrderedDict<std::string, torch::Tensor> get_named_parameters_to_optimise(void) {
            auto ow = optimise_weights.item<bool>();
            auto oc = optimise_components.item<bool>();

            if (!ow && !oc) {
                throw std::logic_error("In the Ensemble ProbabilisticModule, we must optimise at least one of weights and components.");
            }

            auto out = named_parameters(oc);

            if (!ow) {
                out.erase(shapely_parameter_raw_name(weights->name()));
            }

            return out;
        }

        std::vector<torch::Tensor> get_parameters_to_optimise(void) {
            auto ow = optimise_weights.item<bool>();
            auto oc = optimise_components.item<bool>();

            if (!ow && !oc) {
                throw std::logic_error("In the Ensemble ProbabilisticModule, we must optimise at least one of the weights and components.");
            }

            return parameters(oc);
        }

        void estimating_equations_values_recursive_update(
            torch::OrderedDict<std::string, torch::Tensor>& to_update,
            torch::OrderedDict<std::string, torch::Tensor>&& the_update,
            const std::string& nested_module_name
        ) const {
            if (!the_update.is_empty()) {
                auto nested_module_name_dot = nested_module_name + '.';
                to_update.reserve(to_update.size() + the_update.size());
                for (auto& item : the_update) {
                    to_update.insert(nested_module_name_dot + item.key(), std::move(item.value()));
                }
            }
        }

        template<class K1, class K2, class V>
        void estimating_equations_values_recursive_update(
            torch::OrderedDict<K1, torch::OrderedDict<K2, V>>& to_update,
            torch::OrderedDict<K1, torch::OrderedDict<K2, V>>&& the_update,
            const std::string& nested_module_name
        ) const {
            to_update.reserve(to_update.size() + the_update.size());
            for (auto& item : the_update) {
                const auto& key = item.key();
                auto *to_update_value = to_update.find(key);
                if (to_update_value) {
                    estimating_equations_values_recursive_update(
                        *to_update_value,
                        std::move(item.value()),
                        nested_module_name
                    );
                } else {
                    auto& to_update_value = to_update.insert(key, torch::OrderedDict<K2, V>());
                    estimating_equations_values_recursive_update(
                        to_update_value,
                        std::move(item.value()),
                        nested_module_name
                    );
                }
            }
        }

        std::shared_ptr<Simplex> weights;
        torch::OrderedDict<std::string, std::shared_ptr<ProbabilisticModule>> components;

        torch::Tensor optimise_weights;
        torch::Tensor fixed_weights;
        torch::Tensor optimise_components;
        torch::Tensor fixed_components;
};

std::unique_ptr<ProbabilisticModule> ManufactureEnsemble(
    std::vector<std::shared_ptr<ProbabilisticModule>> components,
    NamedShapelyParameters& shapely_parameters,
    Buffers& buffers
) {
    return Ensemble::FromNamedShapelyParameters(
        std::move(components),
        shapely_parameters,
        buffers
    );
}

std::shared_ptr<ProbabilisticModule> change_components(
    const ProbabilisticModule& ensemble,
    const std::vector<std::shared_ptr<ProbabilisticModule>>& new_components
) {
    torch::OrderedDict<std::string, torch::Tensor> new_parameters;
    auto new_components_size = new_components.size();
    for (decltype(new_components_size) i = 0; i != new_components_size; ++i) {
        const auto& new_component_i = new_components[i];
        auto component_name_i = get_component_name(i);
        auto parameters_i = new_component_i->named_parameters();
        new_parameters.reserve(new_parameters.size() + parameters_i.size());
        for (const auto& item : parameters_i) {
            new_parameters.insert(component_name_i + "." + item.key(), std::move(item.value()));
        }
    }
    auto new_ensemble = ensemble.clone_probabilistic_module();
    new_ensemble->set_parameters(new_parameters);
    return new_ensemble;
}

