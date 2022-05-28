#ifndef PROBABILISTIC_MODELLING_INFERENCE_SAMPLING_DISTRIBUTION_HPP_GUARD
#define PROBABILISTIC_MODELLING_INFERENCE_SAMPLING_DISTRIBUTION_HPP_GUARD

// Pre-declare below-defined classes, since this header
// includes ProbabilisticModule.hpp that includes this header.
class SamplingDistribution;

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/score/ScoringRule.hpp>

class SamplingDistribution {
    public:
        virtual std::shared_ptr<ProbabilisticModule> get_fit(void) const {
            throw std::runtime_error("SamplingDistribution::get_fit unimplemented.");
        }

        virtual const ProbabilisticModule& get_fit_ref(void) const {
            throw std::runtime_error("SamplingDistribution::get_fit_ref unimplemented.");
        }

        virtual std::shared_ptr<Distribution> get_centered_parameter_estimate_distribution(void) const {
            throw std::runtime_error("SamplingDistribution::get_centered_parameter_estimate_distribution unimplemented.");
        }

        virtual std::shared_ptr<Distribution> get_parameter_distribution(void) const {
            throw std::runtime_error("SamplingDistribution::get_parameter_estimate_distribution unimplemented.");
        }

        virtual std::shared_ptr<Distribution> get_performance_divergence_distribution(void) const;

        virtual std::shared_ptr<Distribution> get_performance_divergence_distribution(const ScoringRule& scoring_rule) const;

        virtual std::shared_ptr<ProbabilisticModule> draw_stochastic_process(void) const; 

        std::unique_ptr<Distribution> get_centered_function_estimate_distribution(
            std::string function_name,
            torch::OrderedDict<std::string, torch::Tensor> jac,
            torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> hess
        ) const;

        std::unique_ptr<Distribution> get_centered_function_estimate_distribution(
            std::string function_name,
            std::function<torch::Tensor(ProbabilisticModule&)> f
        ) const;

        std::unique_ptr<Distribution> get_function_distribution(
            std::string function_name,
            torch::Tensor function_value,
            torch::OrderedDict<std::string, torch::Tensor> jac,
            torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> hess
        ) const;

        std::unique_ptr<Distribution> get_function_distribution(
            std::string function_name,
            std::function<torch::Tensor(ProbabilisticModule&)> f
        ) const;

        virtual ~SamplingDistribution() { }
};

std::unique_ptr<Distribution> get_centered_function_estimate_distribution(
    std::string function_name,
    std::shared_ptr<Distribution> centered_parameter_estimate_distribution,
    torch::OrderedDict<std::string, torch::Tensor> jac,
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> hess
);

#endif

