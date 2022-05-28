#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <libtorch_support/time_series.hpp>
#include <modelling/sample_size.hpp>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/score/ScoringRule.hpp>

torch::OrderedDict<std::string, torch::Tensor> get_scores_not_na(const torch::OrderedDict<std::string, torch::Tensor>& scores) {
    torch::OrderedDict<std::string, torch::Tensor> scores_not_na;
    scores_not_na.reserve(scores.size());
    for (const auto& item : scores) {
        scores_not_na.insert(
            item.key(),
            missing::isna(item.value()).logical_not()
        );
    }
    return scores_not_na;    
}

torch::OrderedDict<std::string, torch::Tensor> ScoringRule::score(
    const Distribution& forecasts,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    const torch::OrderedDict<std::string, torch::Tensor>& barrier
) const {
    auto score_no_barrier = score(forecasts, observations);
    torch::OrderedDict<std::string, torch::Tensor> score_with_barrier; score_with_barrier.reserve(score_no_barrier.size());
    for (const auto& item : score_no_barrier) {
        const auto& key = item.key();
        score_with_barrier.insert(
            key,
            missing::handle_na(
                [](const torch::Tensor& score_no_barrier_i, const torch::Tensor& barrier_i) {
                    return score_no_barrier_i + barrier_i;
                },
                item.value(),
                barrier[key]
            )
        );
    }
    return score_with_barrier;
}

torch::Tensor sum(
    const torch::OrderedDict<std::string, torch::Tensor>& scores,
    const torch::OrderedDict<std::string, torch::Tensor>& scores_not_na
) {
    torch::Tensor sum_scores = torch::full({}, 1, torch::kDouble);
    for (const auto& item : scores) {
        sum_scores += item.value().masked_select(scores_not_na[item.key()]).sum();
    }
    return sum_scores;
}

torch::Tensor sum(const torch::OrderedDict<std::string, torch::Tensor>& scores) {
    auto scores_not_na = get_scores_not_na(scores);
    return sum(scores, scores_not_na);
}

torch::Tensor ScoringRule::sum(const torch::OrderedDict<std::string, torch::Tensor>& scores) const {
    return ::sum(scores);
}

torch::Tensor ScoringRule::sum(
    const Distribution& forecasts,
    const torch::OrderedDict<std::string, torch::Tensor>& observations
) const {
    return sum(score(forecasts, observations));
}

torch::Tensor ScoringRule::sum(
    const Distribution& forecasts,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    const torch::OrderedDict<std::string, torch::Tensor>& barrier
) const {
    return sum(score(forecasts, observations, barrier));
}

torch::Tensor average(
    const torch::OrderedDict<std::string, torch::Tensor>& scores,
    const torch::OrderedDict<std::string, torch::Tensor>& scores_not_na,
    int64_t sample_size
) {
    return sum(scores, scores_not_na)/sample_size;
}

torch::Tensor average(const torch::OrderedDict<std::string, torch::Tensor>& scores, int64_t sample_size) {
    auto scores_not_na = get_scores_not_na(scores);
    return average(scores, scores_not_na, sample_size);
}

torch::Tensor ScoringRule::average(const torch::OrderedDict<std::string, torch::Tensor>& scores) const {
    auto scores_not_na = get_scores_not_na(scores);
    auto sample_size = get_sample_size(scores_not_na);
    return ::average(scores, scores_not_na, sample_size);
}

torch::Tensor ScoringRule::average(
    const Distribution& forecasts,
    const torch::OrderedDict<std::string, torch::Tensor>& observations
) const {
    /*
    for (const auto& item : observations) {
        auto ndim = item.value().ndimension();

        // Assume that the first dim-1 dimensions correspond to
        // the dimensionality of the index-set of the stochastic
        // process (e.g. 1 for time series, 2 for spatial, 3 for
        // spatio-temporal), and that the index of the last
        // dimension is the index of the coordinate, which we
        // assume is a (possibly one dimensional) vector.
        // See the first page of the Stochastic Processes
        // chapter of Stochastic Limit Theory by James Davidson
        // for an introduction to the terms "index-set" and
        // "coordinate". We assume that the index-set is N^k
        // for some finitek. Accordingly,
        if (ndim < 2) {
            std::ostringstream ss;
            ss << "ScoringRule::average: ndim = " << ndim << " < 2.";
            throw std::logic_error(ss.str());
        }
    }
    */
    // I think the above will become the pervue of the Distribution, and
    // not to be imposed library wide. Will leave this (and the below)
    // commented out for now while I mull it over.

    return average(score(forecasts, observations));
}

torch::Tensor ScoringRule::average(
    const Distribution& forecasts,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    int64_t sample_size
) const {
    /*
    for (const auto& item : observations) {
        auto ndim = item.value().ndimension();
        if (ndim < 2) {
            std::ostringstream ss;
            ss << "ScoringRule::average: ndim = " << ndim << " < 2.";
            throw std::logic_error(ss.str());
        }
    }
    */
    auto scores = score(forecasts, observations);
    return ::average(scores, sample_size);
}

torch::Tensor ScoringRule::average(
    const Distribution& forecasts,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    const torch::OrderedDict<std::string, torch::Tensor>& barrier
) const {
    return average(score(forecasts, observations, barrier));
}

torch::Tensor ScoringRule::average(
    const Distribution& forecasts,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    const torch::OrderedDict<std::string, torch::Tensor>& barrier,
    int64_t sample_size
) const {
    auto scores = score(forecasts, observations, barrier);
    return ::average(scores, sample_size);
}

torch::Tensor ScoringRule::average_out_of_sample(
    const Distribution& forecasts,
    const torch::OrderedDict<std::string, torch::Tensor>& observations,
    int64_t in_sample_times
) const {
    SampleSplitter splitter(in_sample_times);
    auto all_scores = score(forecasts, observations);
    auto out_of_sample_scores = splitter.out_of_sample(all_scores);
    return average(out_of_sample_scores);
}

/*
bool ScoringRule::operator==(const ScoringRule& rhs) const {
    throw std::runtime_error("bool ScoringRule::operator==(const ScoringRule&) unimplemented.");
}
*/

