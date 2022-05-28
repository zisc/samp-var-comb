#ifndef PROBABILISTIC_CENSOREDLOGSCORE_HPP_GUARD
#define PROBABILISTIC_CENSOREDLOGSCORE_HPP_GUARD

#include <memory>
#include <modelling/score/ScoringRule.hpp>

// The CensoredLogScore is a proper scoring rule that focuses
// on the performance of forecast distributions on the set
// (open_lower_bound, closed_lower_bound] if complement = false, or
// its complement if complement = true.
std::unique_ptr<ScoringRule> ManufactureCensoredLogScore(
    double open_lower_bound,
    double closed_upper_bound,
    bool complement = false
);

// The ProbabilityCensoredLogScore is a proper scoring rule that
// focuses on the performance of the forecast distributions on the set
// (open_lower_probability^th percentile, upper_closed_probability^th percentile]
// if complement = false, or its complement if complement = true.
std::unique_ptr<ScoringRule> ManufactureProbabilityCensoredLogScore(
    double open_lower_probability,
    double closed_upper_probability,
    bool complement
);

#endif

