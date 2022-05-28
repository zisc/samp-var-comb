#include <string>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/score/LogScore.hpp>

class LogScore : public ScoringRule {
    public:
        virtual std::string name(void) const override {
            return "LogScore";
        }

        virtual torch::OrderedDict<std::string, torch::Tensor> score(
            const Distribution& forecasts,
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            return forecasts.log_density(observations);
        }
        
        /*
        virtual bool operator==(const ScoringRule& rhs) const override {
            return dynamic_cast<const LogScore*>(&rhs);
        }
        */
};

std::unique_ptr<ScoringRule> ManufactureLogScore(void) {
    return std::make_unique<LogScore>();
}

