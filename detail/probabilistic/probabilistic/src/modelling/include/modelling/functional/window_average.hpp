#ifndef PROBABILISTIC_MODELLING_FUNCTIONAL_EXPANDING_WINDOW_AVERAGE
#define PROBABILISTIC_MODELLING_FUNCTIONAL_EXPANDING_WINDOW_AVERAGE

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <libtorch_support/time_series.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/score/ScoringRule.hpp>
#include <torch/torch.h>

torch::Tensor out_of_sample_average(
    std::function<torch::Tensor(const Distribution&, const torch::OrderedDict<std::string, torch::Tensor>&, const SampleSplitter&)> functional,
    std::shared_ptr<ProbabilisticModule> model,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    int64_t in_sample_times,
    int64_t time_dimension = -1
);

torch::Tensor out_of_sample_average(
    std::function<torch::Tensor(const Distribution&, const torch::OrderedDict<std::string, torch::Tensor>&, const SampleSplitter&)> functional,
    int64_t in_sample_times,
    int64_t time_dimension = -1
);

torch::Tensor expanding_window_average(
    std::function<torch::Tensor(const Distribution&, const torch::OrderedDict<std::string, torch::Tensor>&, const SampleSplitter&)> functional,
    std::shared_ptr<ProbabilisticModule> model,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    int64_t min_in_sample_times,
    int64_t time_dimension = -1
);

torch::Tensor expanding_window_average(
    std::function<torch::Tensor(const Distribution&, const torch::OrderedDict<std::string, torch::Tensor>&, const SampleSplitter&)> functional,
    std::shared_ptr<ProbabilisticModule> model,
    int64_t min_in_sample_times,
    int64_t time_dimension = -1
);

#endif

