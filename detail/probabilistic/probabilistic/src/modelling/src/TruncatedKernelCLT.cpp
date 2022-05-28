#include <cmath>
#include <memory>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <libtorch_support/indexing.hpp>
#include <libtorch_support/moments.hpp>
#include <libtorch_support/missing.hpp>
#include <modelling/sample_size.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/NormalVector.hpp>
#include <modelling/distribution/Quadratic.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/inference/SamplingDistribution.hpp>
#include <modelling/inference/TruncatedKernelCLT.hpp>

#include <log/trivial.hpp>

class TruncatedKernelCLT : public SamplingDistribution {
    public:
        TruncatedKernelCLT(
            std::shared_ptr<ProbabilisticModule> fit_in,
            std::shared_ptr<Distribution> centered_parameter_estimate_distribution_in,
            std::shared_ptr<Distribution> parameter_distribution_in,
            std::shared_ptr<Distribution> performance_divergence_distribution_in = nullptr
        ):
            fit(std::move(fit_in)),
            centered_parameter_estimate_distribution(std::move(centered_parameter_estimate_distribution_in)),
            parameter_distribution(std::move(parameter_distribution_in)),
            performance_divergence_distribution(std::move(performance_divergence_distribution_in))
        { }

        std::shared_ptr<ProbabilisticModule> get_fit(void) const override {
            return fit->clone_probabilistic_module();
        }

        const ProbabilisticModule& get_fit_ref(void) const override {
            return *fit;
        }

        std::shared_ptr<Distribution> get_centered_parameter_estimate_distribution(void) const override {
            return centered_parameter_estimate_distribution;
        }

        std::shared_ptr<Distribution> get_parameter_distribution(void) const override {
            return parameter_distribution;
        }

        std::shared_ptr<Distribution> get_performance_divergence_distribution(void) const override {
            return performance_divergence_distribution;
        }

        std::shared_ptr<ProbabilisticModule> draw_stochastic_process(void) const override{
            return fit->draw_stochastic_process(*parameter_distribution);
        }

    private:
        std::shared_ptr<ProbabilisticModule> fit;
        std::shared_ptr<Distribution> centered_parameter_estimate_distribution;
        std::shared_ptr<Distribution> parameter_distribution;
        std::shared_ptr<Distribution> performance_divergence_distribution;
};

std::shared_ptr<SamplingDistribution> ManufactureTruncatedKernelCLT(
    std::shared_ptr<ProbabilisticModule> fit,
    int64_t dependent_index
) {
    auto& model = *fit;

    PROBABILISTIC_LOG_TRIVIAL_INFO << "Begin TruncatedKernelCLT estimation of the sampling distribution for the parameter estimates of model \"" << model.name() << "\".";

    const auto& observations = model.observations();
    const auto& scoring_rule = *model.scoring_rule();
    auto barrier_multiplier = model.barrier_multiplier();
    auto model_forward = model.forward(observations);
    auto model_barrier = model.barrier(observations, barrier_multiplier);

    auto scores = scoring_rule.score(
        *model_forward,
        observations,
        model_barrier
    );
    
    auto total_score = scoring_rule.sum(scores);

    auto full_sample_size = get_sample_size(scores);

    auto parameters = model.named_parameters(/*recurse=*/true, /*include_fixed=*/false);

    auto estimating_equations_values = fit->estimating_equations_values(/*create_graph=*/true);

    torch::OrderedDict<std::string, torch::Tensor> estimating_equations_sums = [&]() {
        std::vector<int64_t> observation_indices;
        auto estimating_equations_sums_each_series = elementwise_unary_op(
            estimating_equations_values,
            [&observation_indices](const torch::Tensor& x) {
                auto observation_indices_size = x.sizes().size()-1;
                if (observation_indices.size() != observation_indices_size) {
                    observation_indices.clear();
                    observation_indices.reserve(observation_indices_size);
                    for (int64_t i = 0; i != observation_indices_size; ++i) {
                        observation_indices.emplace_back(i);
                    }
                }
                return missing::replace_na(x, 0.0).sum(observation_indices);
            },
            /*missing_participates=*/true
        ).values();
        return std::accumulate(
            estimating_equations_sums_each_series.begin()+1,
            estimating_equations_sums_each_series.end(),
            estimating_equations_sums_each_series.front()
        );
    }();

    auto observations_by_parameter = model.observations_by_parameter(observations);

    auto ssbp = sample_size_by_element(estimating_equations_values, observations_by_parameter/*, 0.0*/).values();
    torch::OrderedDict<std::string, torch::Tensor> sample_size_by_parameter = std::accumulate(
        ssbp.begin()+1,
        ssbp.end(),
        ssbp.front()
    );

    // In this line, observations_by_parameter is moved from.
    auto observations_by_parameterisation = partition(std::move(observations_by_parameter), sizes(observations));

    auto sample_size_by_parameterisation = sample_size(estimating_equations_values, observations_by_parameterisation);

    auto estimating_equations_asymptotic_variance = truncated_kernel_asymptotic_covariance_matrix_panel(
        estimating_equations_values,
        sample_size_by_parameter,
        observations_by_parameterisation,
        sample_size_by_parameterisation,
        dependent_index// ,
        // 0.0
    );

    auto total_score_jacobian = jacobian(total_score, parameters, JacobianMode::Auto, /*create_graph=*/true);
    auto average_score_jacobian = total_score_jacobian/full_sample_size;
    auto average_score_hessian = jacobian(average_score_jacobian, parameters);
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> estimating_equations_jacobian = jacobian(
        estimating_equations_sums/sample_size_by_parameter,
        parameters,
        JacobianMode::Auto,
        /*create_graph=*/false,
        /*allow_unused=*/true
    );

    auto sample_size_by_parameter_collapsed = collapse_vector(sample_size_by_parameter);
    auto ssbpc_size = sample_size_by_parameter_collapsed.tensor.sizes().at(0);
    auto sqrt_sample_size_by_parameter = sample_size_by_parameter_collapsed.tensor
                                                    .toType(torch::kDouble)
                                                    .sqrt()
                                                    .unsqueeze(1)
                                                    .expand({ssbpc_size, ssbpc_size});

    auto estimating_equations_asymptotic_variance_collapsed = collapse_matrix(estimating_equations_asymptotic_variance);

    auto estimating_equations_jacobian_collapsed = collapse_matrix(estimating_equations_jacobian);

    auto sqrt_parameters_asymptotic_variance = [&]() {
        try {
            auto eig = estimating_equations_asymptotic_variance_collapsed.tensor.symeig(true);
            auto sqrt_asy_var = torch::matmul(std::get<1>(eig), std::get<0>(eig).sqrt().diag());
            return std::get<0>(sqrt_asy_var.solve(estimating_equations_jacobian_collapsed.tensor));
        } catch (...) {
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << "estimating_equations_asymptotic_variance\n\n";
            for (const auto& item_i : estimating_equations_asymptotic_variance) {
                const auto& key_i = item_i.key();
                const auto& eeav_i = item_i.value();
                for (const auto& item_ij : eeav_i) {
                    const auto& key_j = item_ij.key();
                    const auto& eeav_ij = item_ij.value();
                    PROBABILISTIC_LOG_TRIVIAL_DEBUG << "cov(" << key_i << ", " << key_j << ")\n\n"
                                                    << eeav_ij << "\n\n";
                }
            }

            PROBABILISTIC_LOG_TRIVIAL_DEBUG << "estimating_equations_jacobian\n\n";
            for (const auto& item_i : estimating_equations_jacobian) {
                const auto& key_i = item_i.key();
                const auto& eej_i = item_i.value();
                for (const auto& item_ij : eej_i) {
                    const auto& key_j = item_ij.key();
                    const auto& eej_ij = item_ij.value();
                    PROBABILISTIC_LOG_TRIVIAL_DEBUG << "hess(" << key_i << ", " << key_j << ")\n\n"
                                                    << eej_ij << "\n\n";
                }
            }

            PROBABILISTIC_LOG_TRIVIAL_DEBUG << "estimating_equations_asymptotic_variance_collapsed\n\n"
                                            << estimating_equations_asymptotic_variance_collapsed.tensor << "\n\n"
                                               "estimating_equations_jacobian_collapsed\n\n"
                                            << estimating_equations_jacobian_collapsed.tensor << "\n\n";

            throw;
        }
    }();

    auto parameters_collapsed = collapse_vector(parameters);
    
    auto sqrt_parameters_asymptotic_variance_on_sqrt_n = sqrt_parameters_asymptotic_variance / sqrt_sample_size_by_parameter;

    auto parameter_distribution = ManufactureNormalVectorDetail(
        parameters_collapsed.tensor,
        sqrt_parameters_asymptotic_variance_on_sqrt_n,
        parameters_collapsed.indices
    );

    auto negative_average_score_hessian = -average_score_hessian;

    std::shared_ptr<Distribution> centered_parameter_estimate_distribution = ManufactureNormalVectorDetail(
        torch::full(parameters_collapsed.tensor.sizes(), 0.0, torch::kDouble),
        sqrt_parameters_asymptotic_variance_on_sqrt_n,
        parameters_collapsed.indices
    );

    auto performance_divergence_distribution = get_centered_function_estimate_distribution(
        "Expected Score Divergence",
        centered_parameter_estimate_distribution,
        -average_score_jacobian,
        negative_average_score_hessian
    );

    auto ret = std::make_unique<TruncatedKernelCLT>(
        std::move(fit),
        std::move(centered_parameter_estimate_distribution),
        std::move(parameter_distribution),
        std::move(performance_divergence_distribution)
    );

    PROBABILISTIC_LOG_TRIVIAL_INFO << "End TruncatedKernelCLT estimation.";

    return ret;
}

