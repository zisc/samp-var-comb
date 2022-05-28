#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <libtorch_support/indexing.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/Quadratic.hpp>
#include <modelling/model/ProbabilisticModule.hpp>

inline void throw_if_multi_valued(const torch::Tensor& f_fit) {
    if (f_fit.numel() != 1) {
        std::ostringstream ss;
        ss << "SamplingDistribution supports getting the distribution of functions f with scalar return values, but f(fit).numel() == " << f_fit.numel() << '.';
        throw std::runtime_error(ss.str());
    }
}

std::unique_ptr<Distribution> get_centered_function_estimate_distribution(
    std::string function_name,
    std::shared_ptr<Distribution> centered_parameter_estimate_distribution,
    torch::OrderedDict<std::string, torch::Tensor> jac,
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> hess
) {
    for (auto& item : jac) {
        auto& jac_i = item.value();
        jac_i = jac_i.detach();
    }

    for (auto& item_i : hess) {
        auto& hess_i = item_i.value();
        for (auto& item_j : hess_i) {
            auto& hess_ij = item_j.value();
            hess_ij = hess_ij.detach();
        }
    }

    return ManufactureQuadratic(
        std::move(function_name),
        centered_parameter_estimate_distribution,
        0.0,
        std::move(jac),
        0.5*hess
    );
}

std::shared_ptr<ProbabilisticModule> SamplingDistribution::draw_stochastic_process(void) const {
    return get_fit()->draw_stochastic_process(*get_parameter_distribution());
}

std::unique_ptr<Distribution> SamplingDistribution::get_centered_function_estimate_distribution(
    std::string function_name,
    torch::OrderedDict<std::string, torch::Tensor> jac,
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> hess
) const {
    return ::get_centered_function_estimate_distribution(
        std::move(function_name),
        get_centered_parameter_estimate_distribution(),
        std::move(jac),
        std::move(hess)
    );
}

std::unique_ptr<Distribution> SamplingDistribution::get_centered_function_estimate_distribution(
    std::string function_name,
    std::function<torch::Tensor(ProbabilisticModule&)> f
 ) const {
    auto fit = get_fit();
    auto f_fit = f(*fit); throw_if_multi_valued(f_fit);
    auto params = fit->named_parameters(/*recurse=*/true, /*include_fixed=*/false);
    auto jac = jacobian(f_fit, params, JacobianMode::Auto, /*create_graph=*/true);
    auto hess = jacobian(jac, params);
    return get_centered_function_estimate_distribution(
        std::move(function_name),
        std::move(jac),
        std::move(hess)
    );
}

std::unique_ptr<Distribution> SamplingDistribution::get_function_distribution(
    std::string function_name,
    torch::Tensor function_value,
    torch::OrderedDict<std::string, torch::Tensor> jac,
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> hess
) const {
    throw_if_multi_valued(function_value);

    for (auto& item : jac) {
        auto& jac_i = item.value();
        jac_i = jac_i.detach();
    }

    for (auto& item_i : hess) {
        auto& hess_i = item_i.value();
        for (auto& item_j : hess_i) {
            auto& hess_ij = item_j.value();
            hess_ij = hess_ij.detach();
        }
    }

    return ManufactureQuadratic(
        std::move(function_name),
        get_centered_parameter_estimate_distribution(),
        std::move(function_value),
        -jac,
        -hess
    );
}

std::unique_ptr<Distribution> SamplingDistribution::get_function_distribution(
    std::string function_name,
    std::function<torch::Tensor(ProbabilisticModule&)> f
) const {
    auto fit = get_fit();
    auto f_fit = f(*fit); throw_if_multi_valued(f_fit);
    auto params = fit->named_parameters(/*recurse=*/true, /*include_fixed=*/false);
    auto jac = jacobian(f_fit, params, JacobianMode::Auto, /*create_graph=*/true);
    auto hess = jacobian(jac, params);
    for (auto& item : jac) { auto& j = item.value(); j = j.detach(); }
    return get_function_distribution(
        std::move(function_name),
        std::move(f_fit),
        -jac,
        -hess
    );
}

// We need to think through how exactly we want the barrier to be a part of getting the performance divergence.
// In fact, I think that we need to remove the barrier component of this.

std::shared_ptr<Distribution> SamplingDistribution::get_performance_divergence_distribution(void) const {
    return get_centered_function_estimate_distribution(
        "Expected Score",
        [](ProbabilisticModule& fit) {
            const auto& observations = fit.observations();
            const auto& scoring_rule = fit.scoring_rule();
            // auto barrier_multiplier = fit.barrier_multiplier();
            return scoring_rule->average(
                *(fit.forward(observations)),
                observations//,
                // fit.barrier(observations, barrier_multiplier)
            );
        }
    );
}

std::shared_ptr<Distribution> SamplingDistribution::get_performance_divergence_distribution(const ScoringRule& scoring_rule) const {
    return get_centered_function_estimate_distribution(
        "Expected Score",
        [&scoring_rule](ProbabilisticModule& fit) {
            const auto& observations = fit.observations();
            return scoring_rule.average(
                *(fit.forward(observations)),
                observations
            );
        }
    );
}

