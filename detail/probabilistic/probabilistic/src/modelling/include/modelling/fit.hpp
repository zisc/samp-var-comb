#ifndef PROBABILISTIC_FIT_HPP_GUARD
#define PROBABILISTIC_FIT_HPP_GUARD

// Pre-declare FitDiagnostics and FitPlan, since this header
// includes ProbabilisticModule.hpp, which includes this header.
struct FitDiagnostics;

struct FitPlan {
    double barrier_begin = 1.0;
    double barrier_end = 1e-6;
    double barrier_decay = 0.99;
    double learning_rate = 0.02;
    double tolerance_grad = 0.0;
    double tolerance_change = 0.0;
    int64_t maximum_optimiser_iterations = 10000;
    int64_t timeout_in_seconds = 600;
};

inline FitPlan null_fit_plan(void) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan, nan, nan, nan, -1, -1};
}

inline bool is_null(const FitPlan& plan) {
    return std::isnan(plan.barrier_begin);
}

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <torch/torch.h>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/score/ScoringRule.hpp>

struct FitDiagnostics {
    int64_t seconds;
    std::vector<double> score;
    std::vector<torch::OrderedDict<std::string, torch::Tensor>> parameters;
    std::vector<torch::OrderedDict<std::string, torch::Tensor>> gradient;

    FitDiagnostics clone(void) {
        auto clone_vec_dict = [](const std::vector<torch::OrderedDict<std::string, torch::Tensor>> in) {
            auto out = in;
            for (auto& dict : out) {
                for (auto& item : dict) {
                    item.value() = item.value().clone();
                }
            }
            return out;
        };

        FitDiagnostics out = {
            seconds,
            score,
            clone_vec_dict(parameters),
            clone_vec_dict(gradient)
        };

        return out;
    }
};

void append_to_fit_diagnostics(
    FitDiagnostics* fit_diagnostics,
    const torch::Tensor& loss,
    const ProbabilisticModule& module
);

void pad_fit_diagnostics(FitDiagnostics* fit_diagnostics);

std::shared_ptr<ProbabilisticModule> fit(
    std::shared_ptr<ProbabilisticModule> model,
    std::shared_ptr<const torch::OrderedDict<std::string, torch::Tensor>> observations,
    std::shared_ptr<const ScoringRule> score,
    const FitPlan& plan = FitPlan(),
    FitDiagnostics *diagnostics = nullptr,
    bool *success = nullptr
);

std::shared_ptr<ProbabilisticModule> fit(
    std::shared_ptr<ProbabilisticModule> model,
    std::shared_ptr<const torch::OrderedDict<std::string, torch::Tensor>> observations,
    std::shared_ptr<const ScoringRule> score,
    double barrier_begin = 1.0,
    double barrier_end = 1e-6,
    double barrier_decay = 0.99,
    int64_t maximum_optimiser_iterations = 10000,
    int64_t timeout_in_seconds = 600,
    torch::optim::LBFGSOptions optimiser_options = {},
    FitDiagnostics *diagnostics = nullptr,
    bool *success = nullptr
);

#endif

