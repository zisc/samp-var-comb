#include <cmath>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/score/CensoredLogScore.hpp>

// The CensoredLogScore is a proper scoring rule that focuses
// on the performance of forecast distributions on the set
// (open_lower_bound, closed_upper_bound] if complement = false,
// or its complement if complement = true.

class CensoredLogScore : public ScoringRule {
    public:
        CensoredLogScore(
            double open_lower_bound_in,
            double closed_upper_bound_in,
            bool complement_in
        ):
            open_lower_bound(open_lower_bound_in),
            closed_upper_bound(closed_upper_bound_in),
            complement(complement_in),
            name_store([&]() {
                std::ostringstream ss;
                ss << "CensoredLogScore("
                   << open_lower_bound_in << ", "
                   << closed_upper_bound_in << ", "
                   << (complement_in ? "tails" : "middle") << ")";
                return ss.str();
            }())
        { }

        virtual std::string name(void) const override {
            return name_store;
        }

        virtual torch::OrderedDict<std::string, torch::Tensor> score(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            auto log_score = forecasts.log_density(observations);
            
            auto log_probability_off_focus = [&]() {
                if (complement) {
                    // Focus on complement of interval, so we need log probability of interval.
                    return forecasts.log_interval_probability(open_lower_bound, closed_upper_bound);
                }
                    
                // Focus on interval, so we need the log probability of interval's complement.
                return forecasts.log_interval_complement_probability(open_lower_bound, closed_upper_bound);
            }();

            decltype(log_score) censored_log_score;
            censored_log_score.reserve(log_score.size());
            for (const auto& item : log_score) {
                const auto& name = item.key();
                const auto& log_score_i = item.value();
                const auto& observations_i = observations[name];
                const auto& log_probability_off_focus_i = log_probability_off_focus[name];
                auto observations_in_focus_i = [&]() {
                    if (complement) {
                        return torch::logical_or(observations_i.le(open_lower_bound), observations_i.gt(closed_upper_bound));
                    }

                    return torch::logical_and(observations_i.gt(open_lower_bound), observations_i.le(closed_upper_bound));
                }();

                censored_log_score.insert(name, torch::where(observations_in_focus_i, log_score_i, log_probability_off_focus_i));
            }

            return censored_log_score;
        }

    private:
        double open_lower_bound;
        double closed_upper_bound;
        bool complement;
        std::string name_store;
};

std::unique_ptr<ScoringRule> ManufactureCensoredLogScore(
    double open_lower_bound,
    double closed_upper_bound,
    bool complement
) {
    return std::make_unique<CensoredLogScore>(open_lower_bound, closed_upper_bound, complement);
}

class ProbabilityCensoredLogScore : public ScoringRule {
    public:
        ProbabilityCensoredLogScore(
            double open_lower_probability_in,
            double closed_upper_probability_in,
            bool complement_in
        ):
            open_lower_probability(open_lower_probability_in),
            closed_upper_probability(closed_upper_probability_in),
            complement(complement_in),
            name_store([&]() {
                std::ostringstream ss;
                ss << "CensoredLogScoreQuantile("
                   << open_lower_probability_in << ", "
                   << closed_upper_probability_in << ", "
                   << (complement_in ? "tails" : "middle") << ")";
               return ss.str();
           }())
        { }

        virtual std::string name(void) const override {
            return name_store;
        }

        virtual torch::OrderedDict<std::string, torch::Tensor> score(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            auto log_score = forecasts.log_density(observations);
            auto cdfs = forecasts.cdf(observations);

            auto log_probability_off_focus = [&]() {
                if (complement) {
                    // Focus on complement of interval, so we need log probability of interval.
                    return std::log(closed_upper_probability - open_lower_probability);
                }

                // Focus on interval, so we need the log probability of interval's complement.
                return std::log1p(open_lower_probability - closed_upper_probability);
            }();

            decltype(log_score) censored_log_score;
            censored_log_score.reserve(log_score.size());
            for (const auto& item : log_score) {
                const auto& name = item.key();
                const auto& log_score_i = item.value();
                const auto& cdfs_i = cdfs[name];
                const auto& observations_i = observations[name];
                auto observations_in_focus_i = [&]() {
                    if (complement) {
                        return torch::logical_or(cdfs_i.le(open_lower_probability), cdfs_i.gt(closed_upper_probability));
                    }

                    return torch::logical_and(cdfs_i.gt(open_lower_probability), cdfs_i.le(closed_upper_probability));
                }();

                censored_log_score.insert(name, torch::where(observations_in_focus_i, log_score_i, log_probability_off_focus));
            }

            return censored_log_score;
        }

    private:
        double open_lower_probability;
        double closed_upper_probability;
        bool complement;
        std::string name_store;
};

std::unique_ptr<ScoringRule> ManufactureProbabilityCensoredLogScore(
    double open_lower_probability,
    double closed_upper_probability,
    bool complement
) {
    return std::make_unique<ProbabilityCensoredLogScore>(open_lower_probability, closed_upper_probability, complement);
}

