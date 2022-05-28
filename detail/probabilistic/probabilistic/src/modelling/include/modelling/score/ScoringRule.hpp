#ifndef PROBABILISTIC_SCORINGRULE_HPP_GUARD
#define PROBABILISTIC_SCORINGRULE_HPP_GUARD

// Pre-declare below-defined classes, since this header
// includes ProbabilisticModule.hpp includes this header.
class ScoringRule;

#include <cstdint>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/model/ProbabilisticModule.hpp>

class ScoringRule {
    public:
        virtual std::string name(void) const = 0;

        virtual torch::OrderedDict<std::string, torch::Tensor> score(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const = 0;

        virtual torch::OrderedDict<std::string, torch::Tensor> score(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            const torch::OrderedDict<std::string, torch::Tensor>& barrier
        ) const;

        virtual torch::Tensor sum(const torch::OrderedDict<std::string, torch::Tensor>& scores) const;

        virtual torch::Tensor sum(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const;

        virtual torch::Tensor sum(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            const torch::OrderedDict<std::string, torch::Tensor>& barrier
        ) const ;

        virtual torch::Tensor average(const torch::OrderedDict<std::string, torch::Tensor>& scores) const;

        virtual torch::Tensor average(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const;

        virtual torch::Tensor average(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            int64_t sample_size
        ) const;

        virtual torch::Tensor average(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            const torch::OrderedDict<std::string, torch::Tensor>& barrier
        ) const;

        virtual torch::Tensor average(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            const torch::OrderedDict<std::string, torch::Tensor>& barrier,
            int64_t sample_size
        ) const;

        virtual torch::Tensor average_out_of_sample(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations,
            int64_t in_sample_times
        ) const;

        // virtual bool operator==(const ScoringRule& rhs) const;

        virtual ~ScoringRule() { }
};

#endif

