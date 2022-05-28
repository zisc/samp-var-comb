#ifndef PROBABILISTIC_TICKSCORE_HPP_GUARD
#define PROBABILISTIC_TICKSCORE_HPP_GUARD

#include <memory>
#include <modelling/score/ScoringRule.hpp>

// The TickScore is a proper scoring rule that focuses
// on the performance of forecasts of a single quantile,
// given its associated probability.
std::unique_ptr<ScoringRule> ManufactureTickScore(double probability);

#endif

