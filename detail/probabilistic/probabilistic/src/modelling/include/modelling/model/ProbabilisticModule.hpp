#ifndef PROBABILISTIC_PROBABILISTIC_MODULE_HPP_GUARD
#define PROBABILISTIC_PROBABILISTIC_MODULE_HPP_GUARD

// Pre-declare below-defined classes, since this header
// includes ScoringRule.hpp and SamplingDistribution.hpp,
// which includes this header.
class ProbabilisticModule;

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/Parameterisation.hpp>
#include <modelling/score/ScoringRule.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/model/ShapelyModule.hpp>
#include <modelling/inference/SamplingDistribution.hpp>
#include <modelling/fit.hpp>

class ProbabilisticModule : public virtual ShapelyModule {
    public:
        virtual std::unique_ptr<Distribution> forward(const torch::OrderedDict<std::string, torch::Tensor>& observations) {
            throw std::runtime_error("ProbabilisticModule::forward unimplemented.");
        }

        virtual torch::OrderedDict<std::string, torch::Tensor> draw_observations(int64_t sample_size, int64_t burn_in_size, double first_draw) const {
            throw std::runtime_error("ProbabilisticModule::draw_observations unimplemented.");
        }
 
        virtual torch::OrderedDict<std::string, torch::Tensor> grad(bool recurse = true) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> barrier(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            torch::Tensor scaling = torch::full({1}, 1.0, torch::kDouble)
        ) const {
            throw std::runtime_error("ProbabilisticModule::barrier unimplemented.");
        }

        torch::OrderedDict<std::string, torch::Tensor> barrier(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            double scaling
        ) const {
            return barrier(observations, torch::full({1}, scaling, torch::kDouble));
        }

        virtual bool fit(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            std::shared_ptr<const ScoringRule> scoring_rule,
            const FitPlan& plan,
            FitDiagnostics *diagnostics
        );

        virtual torch::OrderedDict<std::string, torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>>> observations_by_parameter(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            bool recursive = true,
            bool include_fixed = false
        ) const;

        const torch::OrderedDict<std::string, torch::Tensor>& observations(void) const {
            if (!scoring_rule_last_fit) {
                throw std::logic_error("ProbabilisticModule::observations called, but the module has not been fit before.");
            }

            return observations_last_fit;
        }

        std::shared_ptr<const ScoringRule> scoring_rule(void) const {
            if (!scoring_rule_last_fit) {
                throw std::logic_error("ProbabilisticModule::scoring_rule called, but the module has not been fit before.");
            }
            
            return scoring_rule_last_fit;
        }

        double barrier_multiplier(void) const {
            if (!scoring_rule_last_fit) {
                throw std::logic_error("ProbabilisticModule::barrier_multiplier called, but the module has not been fit before.");
            }

            return barrier_multiplier_last_fit;
        }

        FitPlan fit_plan(void) const {
            if (is_null(fit_plan_last_fit)) {
                throw std::logic_error("ProbabilisticModule::fit_plan called, but the module has not been fit before.");
            }

            return fit_plan_last_fit;
        }

        virtual torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> estimating_equations_values(
            bool create_graph = false,
            bool recurse = true
        );

        std::shared_ptr<ProbabilisticModule> draw_stochastic_process(const Distribution& parameter_estimate_distribution) const;

        std::shared_ptr<ProbabilisticModule> clone_probabilistic_module(void) const;

    protected:
        bool fit(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            std::shared_ptr<const ScoringRule> scoring_rule,
            const FitPlan& plan,
            std::vector<torch::Tensor> parameters_to_optimise,
            FitDiagnostics *diagnostics
        );

        bool fit(
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            const FitPlan& plan,
            std::vector<torch::Tensor> parameters_to_optimise,
            std::function<torch::Tensor(double)> score_closure,
            std::string score_name,
            FitDiagnostics *diagnostics
        );

        torch::OrderedDict<std::string, torch::Tensor> observations_last_fit;
        std::shared_ptr<const ScoringRule> scoring_rule_last_fit;
        double barrier_multiplier_last_fit = std::numeric_limits<double>::quiet_NaN();
        FitPlan fit_plan_last_fit = null_fit_plan();
};

template<typename Derived>
class ProbabilisticCloneable : public virtual ProbabilisticModule, public ShapelyCloneable<Derived> { };

#endif

