#include <memory>
#include <string>
#include <libtorch_support/moments.hpp>
#include <modelling/functional/empirical_coverage.hpp>
#include <modelling/functional/window_average.hpp>
#include <modelling/model/ProbabilisticModule.hpp>

torch::OrderedDict<std::string, torch::Tensor> get_covered(
    const Distribution& forecast_distributions,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    double open_lower_probability,
    double closed_upper_probability,
    bool complement
) {
    auto cdf_obs = forecast_distributions.cdf(observations);
    torch::OrderedDict<std::string, torch::Tensor> covered;
    for (const auto& item : cdf_obs) {
        auto cdf_i = item.value(); 
        auto cdf_i_is_na = missing::isna(cdf_i);
        auto cdf_i_not_na = cdf_i_is_na.logical_not();
        auto covered_i_nans_unhandled = [&]() {
            if (complement) {
                return torch::logical_or(cdf_i.le(open_lower_probability), cdf_i.gt(closed_upper_probability));
            }
            return torch::logical_and(cdf_i.gt(open_lower_probability), cdf_i.le(closed_upper_probability));
        }();
        auto covered_i = cdf_i.new_empty(cdf_i.sizes());
        covered_i.index_put_({cdf_i_is_na}, missing::na);
        covered_i.index_put_({cdf_i_not_na}, covered_i_nans_unhandled.index({cdf_i_not_na}).toType(torch::kDouble));
        covered.insert(item.key(), std::move(covered_i));
    }
    return covered;
}

torch::Tensor empirical_coverage(
    std::shared_ptr<ProbabilisticModule> fit,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    double open_lower_probability,
    double closed_upper_probability,
    bool complement,
    int64_t in_sample_times,
    int64_t time_dimension
) {
    SampleSplitter splitter(in_sample_times, time_dimension);
    auto forecast_distributions = fit->forward(observations);
    auto covered = get_covered(*forecast_distributions, observations, open_lower_probability, closed_upper_probability, complement);
    auto oos_covered = splitter.out_of_sample(covered);
    auto avg = average(oos_covered);
    return avg;
}

torch::Tensor empirical_coverage_expanding_window(
    std::shared_ptr<ProbabilisticModule> fit,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    double open_lower_probability,
    double closed_upper_probability,
    bool complement,
    int64_t min_in_sample_times,
    int64_t time_dimension
) {
    return expanding_window_average(
         [&](
             const Distribution& forecast_distributions,
             const torch::OrderedDict<std::string, torch::Tensor>& observations,
             const SampleSplitter& splitter
         ) {
            auto obs_one_step_ahead = splitter.h_steps_ahead(observations, 1);
            auto cdf_one_step_ahead = splitter.h_steps_ahead(forecast_distributions.cdf(observations), 1);
            auto covered = get_covered(forecast_distributions, observations, open_lower_probability, closed_upper_probability, complement);
            auto hsa_covered = splitter.h_steps_ahead(covered, 1);
            auto avg = average(hsa_covered);
            return avg;
        },
        fit,
        observations,
        min_in_sample_times,
        time_dimension
    );
}

torch::Tensor empirical_coverage_expanding_window(
    std::shared_ptr<ProbabilisticModule> fit,
    double open_lower_probability,
    double closed_upper_probability,
    bool complement,
    int64_t min_in_sample_times,
    int64_t time_dimension
) {
    return empirical_coverage_expanding_window(
        fit,
        fit->observations(),
        open_lower_probability,
        closed_upper_probability,
        complement,
        min_in_sample_times,
        time_dimension
    );
}

