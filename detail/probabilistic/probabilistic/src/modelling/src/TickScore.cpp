#include <string>
#include <sstream>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/score/TickScore.hpp>

// The TickScore is a proper scoring rule that focuses
// on the performance of forecasts of a single quantile,
// given its associated probability.

class TickScore : public ScoringRule {
    public:
        TickScore(double probability_in):
            probability(probability_in),
            name_store([&]() {
                std::ostringstream ss;
                ss << "TickScore(" << probability_in << ")";
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
            auto quantile = forecasts.quantile(probability);
            decltype(quantile) tick_score;
            tick_score.reserve(quantile.size());
            for (const auto& item : quantile) {
                const auto& name = item.key();
                const auto& quantile_i = item.value();
                const auto& observations_i = observations[name];
                tick_score.insert(
                    name,
                    missing::handle_na(
                        [&](const auto& obs, const auto& quants) {
                            return (obs - quants)*torch::where(obs.le(quants), 1.0 - probability, -probability);
                        },
                        observations_i,
                        quantile_i
                    )
                );
            }
            return tick_score;
        }

    private:
        double probability;
        std::string name_store;
};

std::unique_ptr<ScoringRule> ManufactureTickScore(double probability) {
    return std::make_unique<TickScore>(probability);
}

