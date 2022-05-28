#ifndef PROBABILISTIC_MODELLING_FUNCTIONAL_EMPIRICAL_COVERAGE_HPP_GUARD
#define PROBABILISTIC_MODELLING_FUNCTIONAL_EMPIRICAL_COVERAGE_HPP_GUARD

#include <memory>
#include <string>
#include <modelling/model/ProbabilisticModule.hpp>

torch::Tensor empirical_coverage(
    std::shared_ptr<ProbabilisticModule> fit,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    double open_lower_probability,
    double closed_upper_probability,
    bool complement,
    int64_t in_sample_times,
    int64_t time_dimension = -1
);

torch::Tensor empirical_coverage_expanding_window(
    std::shared_ptr<ProbabilisticModule> fit,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    double open_lower_probability,
    double closed_upper_probability,
    bool complement,
    int64_t min_in_sample_times,
    int64_t time_dimension = -1
);

torch::Tensor empirical_coverage_expanding_window(
    std::shared_ptr<ProbabilisticModule> fit,
    double open_lower_probability,
    double closed_upper_probability,
    bool complement,
    int64_t min_in_sample_times,
    int64_t time_dimension = -1
);

#endif

