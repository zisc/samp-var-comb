#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <modelling/sample_size.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/score/ScoringRule.hpp>
#include <modelling/inference/SamplingDistribution.hpp>
#include <modelling/model/ProbabilisticModule.hpp>

#include <log/trivial.hpp>

torch::OrderedDict<std::string, torch::Tensor> ProbabilisticModule::grad(bool recurse) const {
    auto params = named_parameters(recurse);
    for (auto& p : params) {
        auto& p_value = p.value();
        auto p_value_grad = p_value.grad();
        if (p_value_grad.numel() == 0) {
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << "p_value_grad.numel() == 0\n"
                                            << p_value << " = params[" << p.key() << "].\n";
        }
        p.value() = p.value().grad();
    }
    return params;
}

bool ProbabilisticModule::fit(
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    std::shared_ptr<const ScoringRule> scoring_rule,
    const FitPlan& plan,
    FitDiagnostics *diagnostics
) {
    return fit(
        observations,
        std::move(scoring_rule),
        plan,
        parameters(),
        diagnostics
    );
}

torch::OrderedDict<std::string, torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>>> ProbabilisticModule::observations_by_parameter(
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    bool recursive,
    bool include_fixed
) const {
    auto parameters = named_parameters(recursive, include_fixed);
    auto parameters_size = parameters.size();
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>>> out;
    out.reserve(observations.size());
    for (const auto& obs_item : observations) {
        std::vector<torch::indexing::TensorIndex> idx(obs_item.value().ndimension(), torch::indexing::Slice());
        torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>> out_nested;
        out_nested.reserve(parameters_size);
        for (const auto& param_item : parameters) {
            out_nested.insert(param_item.key(), {idx});
        }
        out.insert(obs_item.key(), std::move(out_nested));
    }
    return out;
}

torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> ProbabilisticModule::estimating_equations_values(
    bool create_graph,
    bool recurse
) {
    auto obs = observations();
    return jacobian(
        scoring_rule()->score(
            *forward(obs),
            obs,
            barrier(observations_last_fit, barrier_multiplier())
        ),
        named_parameters(recurse, /*false*/ true),
        JacobianMode::Auto,
        create_graph
    );
}

std::shared_ptr<ProbabilisticModule> ProbabilisticModule::draw_stochastic_process(const Distribution& parameter_estimate_distribution) const {
    auto model_clone = clone_probabilistic_module();
    auto p = parameter_estimate_distribution.generate(1, 0, 0.0);
    model_clone->set_parameters(p);
    return model_clone;
}

std::shared_ptr<ProbabilisticModule> ProbabilisticModule::clone_probabilistic_module(void) const {
    auto cloned = std::dynamic_pointer_cast<ProbabilisticModule>(clone());
    cloned->observations_last_fit = observations_last_fit;
    cloned->scoring_rule_last_fit = scoring_rule_last_fit;
    cloned->barrier_multiplier_last_fit = barrier_multiplier_last_fit;
    return cloned;
}

bool ProbabilisticModule::fit(
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    std::shared_ptr<const ScoringRule> scoring_rule,
    const FitPlan& plan,
    std::vector<torch::Tensor> parameters_to_optimise,
    FitDiagnostics *diagnostics
) {
    // Calculating the sample_size is expensive because of missing values.
    // Find the sample_size on the first run of the score_closure, then
    // reuse it for the remainder of the optimisation.
    int64_t sample_size = -1;
    auto ret = fit(
        observations,
        plan,
        std::move(parameters_to_optimise),
        [this, &observations, &scoring_rule, &sample_size] (double barrier_multiplier) {
            auto forecasts = [&]() {
                try {
                    return forward(observations);
                } catch(...) {
                    parameter_dump();
                    throw;
                }
            }();
            if (sample_size >= 0) {
                return scoring_rule->average(
                    *forecasts,
                    observations,
                    barrier(observations, barrier_multiplier),
                    sample_size
                );
            } else {
                auto scores = scoring_rule->score(
                    *forecasts,
                    observations,
                    barrier(observations, barrier_multiplier)
                );
                sample_size = get_sample_size(scores);
                return scoring_rule->average(scores);
            }
        },
        scoring_rule->name(),
        diagnostics
    );
    scoring_rule_last_fit = std::move(scoring_rule);

    return ret;
}

bool ProbabilisticModule::fit(
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    const FitPlan& plan,
    std::vector<torch::Tensor> parameters_to_optimise,
    std::function<torch::Tensor(double)> score_closure,
    std::string score_name,
    FitDiagnostics *diagnostics
) {
    double barrier_begin = plan.barrier_begin;
    double barrier_end = plan.barrier_end;
    double barrier_decay = plan.barrier_decay;
    int64_t maximum_optimiser_iterations = plan.maximum_optimiser_iterations;
    int64_t timeout_in_seconds = plan.timeout_in_seconds;
    
    auto optimiser = [&]() {
        torch::optim::LBFGSOptions lbfgs_options(plan.learning_rate);
        lbfgs_options.line_search_fn("strong_wolfe");
        lbfgs_options.tolerance_grad(plan.tolerance_grad);
        lbfgs_options.tolerance_change(plan.tolerance_change);
        return torch::optim::LBFGS(std::move(parameters_to_optimise), std::move(lbfgs_options));
    }();

    auto model_name = name();
    PROBABILISTIC_LOG_TRIVIAL_INFO << "Begin optimisation of model \"" << model_name << "\". " << (diagnostics ? "Collecting diagnostics." : "Not collecting diagnostics.");
    auto barrier_multiplier = barrier_begin;
    torch::optim::Optimizer::LossClosure loss_closure = [&optimiser, &score_closure, &barrier_multiplier]() {
        optimiser.zero_grad();
        auto loss = -score_closure(barrier_multiplier);
        loss.backward();
        return loss;
    };
    auto t_start = std::chrono::high_resolution_clock::now();
    int64_t seconds_since_start;
    int64_t num_prints = 0;
    bool success = false;
    try {
        auto loss_after_step_prev = optimiser.step(loss_closure); append_to_fit_diagnostics(diagnostics, loss_after_step_prev, *this);
        auto loss_after_step = optimiser.step(loss_closure); append_to_fit_diagnostics(diagnostics, loss_after_step, *this);
        int64_t optimiser_iterations_completed = 2;
        while (true) {
            loss_after_step_prev = loss_after_step;
            barrier_multiplier = std::max(barrier_decay*barrier_multiplier, barrier_end);
            loss_after_step = optimiser.step(loss_closure); append_to_fit_diagnostics(diagnostics, loss_after_step, *this);
            optimiser_iterations_completed += 1;
            auto t_end = std::chrono::high_resolution_clock::now();
            seconds_since_start = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count();
            if (torch::equal(loss_after_step, loss_after_step_prev)) {
                PROBABILISTIC_LOG_TRIVIAL_INFO << "Optimisation of model \"" << model_name << "\""
                                                  " converged with score \"" << score_name << "\" of " << -loss_after_step.item<double>() << ","
                                                  " including a barrier with multiplier " << barrier_multiplier << ","
                                                  " after " << optimiser_iterations_completed << " iterations"
                                                  " and " << seconds_since_start << " seconds.";
                success = true;
                break;
            }
            if (seconds_since_start >= timeout_in_seconds || optimiser_iterations_completed >= maximum_optimiser_iterations) {
                PROBABILISTIC_LOG_TRIVIAL_WARNING << "Optimisation of model \"" << model_name << "\""
                                                     " timed out with score \"" << score_name << "\" of " << -loss_after_step.item<double>() << ","
                                                     " including a barrier with multiplier " << barrier_multiplier << ","
                                                     " after " << optimiser_iterations_completed << " iterations"
                                                     " and " << seconds_since_start << " seconds.";
                break;
            }
            if (seconds_since_start >= 10*num_prints) {
                PROBABILISTIC_LOG_TRIVIAL_INFO << "Score \"" << score_name << "\" is " << -loss_after_step.item<double>() << ","
                                                  " including a barrier with multiplier " << barrier_multiplier << ","
                                                  " after " << optimiser_iterations_completed << " iterations"
                                                  " and " << seconds_since_start << " seconds.";
                ++num_prints;
            }
        }
    } catch (const std::exception& e) {
        PROBABILISTIC_LOG_TRIVIAL_WARNING << "Optimisation of model \"" << model_name << "\""
                                             " failed with C++ exception \"" << e.what() << "\".";
        if (diagnostics) { diagnostics->score.emplace_back(std::numeric_limits<double>::quiet_NaN()); }
    } catch (...) {
        PROBABILISTIC_LOG_TRIVIAL_WARNING << "Optimisation of model \"" << model_name << "\""
                                             " failed with unknown C++ exception.";
        if (diagnostics) { diagnostics->score.emplace_back(std::numeric_limits<double>::quiet_NaN()); }
    }

    pad_fit_diagnostics(diagnostics);
    if (diagnostics) diagnostics->seconds = seconds_since_start;

    observations_last_fit = observations;
    barrier_multiplier_last_fit = barrier_multiplier;
    fit_plan_last_fit = plan;

    return success;
}

