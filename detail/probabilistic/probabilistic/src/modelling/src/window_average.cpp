#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <libtorch_support/moments.hpp>
#include <libtorch_support/time_series.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/functional/window_average.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/score/ScoringRule.hpp>
#include <torch/torch.h>

// Note that in_sample_size currently does not account for missing values.

auto min_out_of_sample_times(
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    int64_t in_sample_size,
    int64_t time_dimension
) {
    int64_t ret = std::numeric_limits<int64_t>::max();
    for (const auto& item : observations) {
        const auto& obs_i_sizes = item.value().sizes();
        auto t_dim = time_dimension > 0 ? time_dimension : obs_i_sizes.size() + time_dimension;
        ret = std::min(ret, item.value().sizes().at(t_dim) - in_sample_size);
    }
    return ret;
}

torch::Tensor out_of_sample_average(
    std::function<torch::Tensor(const Distribution&, const torch::OrderedDict<std::string, torch::Tensor>&, const SampleSplitter&)> functional,
    std::shared_ptr<ProbabilisticModule> model,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    int64_t in_sample_times,
    int64_t time_dimension
) {
    auto out = torch::empty({}, torch::kDouble);
    if (min_out_of_sample_times(observations, in_sample_times, time_dimension) > 0) {
        SampleSplitter splitter(in_sample_times, time_dimension);
        auto functional_result = functional(*model->forward(observations), observations, splitter);
        if (!functional_result.ndimension()) { functional_result = functional_result.expand({1}); }
        out = average(functional_result);
    }
    return out;
}

torch::Tensor out_of_sample_average(
    std::function<torch::Tensor(const Distribution&, const torch::OrderedDict<std::string, torch::Tensor>&, const SampleSplitter&)> functional,
    std::shared_ptr<ProbabilisticModule> model,
    int64_t in_sample_times,
    int64_t time_dimension
) {
    return out_of_sample_average(
        std::move(functional),
        model,
        model->observations(),
        in_sample_times,
        time_dimension
    );
}

torch::Tensor expanding_window_average(
    std::function<torch::Tensor(const Distribution&, const torch::OrderedDict<std::string, torch::Tensor>&, const SampleSplitter&)> functional,
    std::shared_ptr<ProbabilisticModule> model,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    int64_t min_in_sample_times,
    int64_t time_dimension
) {
    model = model->clone_probabilistic_module();

    auto scoring_rule = model->scoring_rule();
    auto fit_plan = model->fit_plan();
    auto fit_lambda = [&](ProbabilisticModule& model, const torch::OrderedDict<std::string, torch::Tensor>& obs) {
        return model.fit(
            obs,
            scoring_rule,
            fit_plan,
            /* diagnostics = */ nullptr
        );
    };

    auto out = torch::empty({}, torch::kDouble);
    if (min_out_of_sample_times(observations, min_in_sample_times, time_dimension) > 0) {
        SampleSplitter splitter(min_in_sample_times, time_dimension);
        fit_lambda(*model, splitter.in_sample(observations));
        out = functional(*model->forward(observations), observations, splitter);
        if (!out.ndimension()) { out = out.expand({1}); }
        for (auto T = min_in_sample_times+1; min_out_of_sample_times(observations, T, time_dimension) > 0; ++T) {
            SampleSplitter splitter(T, time_dimension);
            fit_lambda(*model, splitter.in_sample(observations));
            auto functional_result = functional(*model->forward(observations), observations, splitter);
            if (!functional_result.ndimension()) { functional_result = functional_result.expand({1}); }
            out = torch::cat({out, functional_result});
        }
        out = average(out);
    }

    return out;
}

torch::Tensor expanding_window_average(
    std::function<torch::Tensor(const Distribution&, const torch::OrderedDict<std::string, torch::Tensor>&, const SampleSplitter&)> functional,
    std::shared_ptr<ProbabilisticModule> model,
    int64_t min_in_sample_times,
    int64_t time_dimension
) {
    return expanding_window_average(
        std::move(functional),
        model,
        model->observations(),
        min_in_sample_times,
        time_dimension
    );
}

