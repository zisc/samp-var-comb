#ifndef PROBABILISTIC_LOGSCORE_HPP_GUARD
#define PROBABILISTIC_LOGSCORE_HPP_GUARD

#include <memory>
#include <modelling/score/ScoringRule.hpp>

std::unique_ptr<ScoringRule> ManufactureLogScore(void);

#endif

