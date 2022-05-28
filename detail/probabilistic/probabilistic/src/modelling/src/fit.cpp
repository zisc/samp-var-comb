#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include <log/trivial.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/score/ScoringRule.hpp>
#include <modelling/sample_size.hpp>
#include <modelling/fit.hpp>

torch::OrderedDict<std::string, torch::Tensor> clone_dict(
    torch::OrderedDict<std::string, torch::Tensor> dict
) {
    for (auto& item : dict) {
        item.value() = item.value().clone();
    }
    return dict;
}

void append_to_fit_diagnostics(
    FitDiagnostics* diagnostics,
    const torch::Tensor& loss,
    const ProbabilisticModule& model
) {
    if (diagnostics) {
        diagnostics->score.emplace_back(-loss.item<double>());
        diagnostics->parameters.emplace_back(clone_dict(model.named_parameters_on_paper()));
        diagnostics->gradient.emplace_back(clone_dict(model.grad()));
    }
}

void pad_fit_diagnostics(FitDiagnostics* diagnostics) {
    if (diagnostics) {
        auto max_size = std::max({diagnostics->score.size(), diagnostics->parameters.size(), diagnostics->gradient.size()});
        while (diagnostics->score.size() < max_size) {
            diagnostics->score.emplace_back(std::numeric_limits<double>::quiet_NaN());
        }
        if (diagnostics->parameters.size() < max_size) {
            auto back = diagnostics->parameters.back();
            do {
                diagnostics->parameters.emplace_back(back);
            } while (diagnostics->parameters.size() < max_size);
        }
        if (diagnostics->gradient.size() < max_size) {
            auto back = diagnostics->gradient.back();
            do {
                diagnostics->gradient.emplace_back(back);
            } while (diagnostics->parameters.size() < max_size);
        }
    }
}

std::shared_ptr<ProbabilisticModule> fit(
    std::shared_ptr<ProbabilisticModule> model,
    std::shared_ptr<const torch::OrderedDict<std::string, torch::Tensor>> observations,
    std::shared_ptr<const ScoringRule> scoring_rule,
    const FitPlan& plan,
    FitDiagnostics *diagnostics,
    bool *success
) {
    model = model->clone_probabilistic_module();

    bool success_nested = model->fit(
        *observations,
        std::move(scoring_rule),
        plan,
        diagnostics
    );
    if (success) { *success = success_nested; }

    return model;
}

std::shared_ptr<ProbabilisticModule> fit(
    std::shared_ptr<ProbabilisticModule> model,
    std::shared_ptr<const torch::OrderedDict<std::string, torch::Tensor>> observations,
    std::shared_ptr<const ScoringRule> scoring_rule,
    double barrier_begin,
    double barrier_end,
    double barrier_decay,
    int64_t maximum_optimiser_iterations,
    int64_t timeout_in_seconds,
    torch::optim::LBFGSOptions optimiser_options,
    FitDiagnostics *diagnostics,
    bool *success
) {
    FitPlan plan = {
        barrier_begin,
        barrier_end,
        barrier_decay,
        optimiser_options.lr(),
        optimiser_options.tolerance_grad(),
        optimiser_options.tolerance_change(),
        maximum_optimiser_iterations,
        timeout_in_seconds
    };

    return fit(
        std::move(model),
        std::move(observations),
        std::move(scoring_rule),
        plan,
        diagnostics,
        success
    );
}

