#ifndef PROBABILISTIC_AUTOREGRESSIVE_HPP_GUARD
#define PROBABILISTIC_AUTOREGRESSIVE_HPP_GUARD

#include <memory>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/Parameterisation.hpp>
#include <modelling/model/ShapelyModule.hpp>

torch::Tensor AutoRegressive_forward(
    torch::Tensor x,
    torch::Tensor coefficients_get
);

torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>> AutoRegressive_observations_by_parameter(
    const torch::Tensor& x,
    const std::shared_ptr<Parameterisation>& coefficients,
    bool recursive
);

// See https://stackoverflow.com/questions/17495763/not-declared-in-this-scope-error-resolving-name-of-function-template-in-a-publ#17495820
// for uses of "template" keyword.

template<class CoefficientsParameterisation>
class AutoRegressive : public ShapelyCloneable<AutoRegressive<CoefficientsParameterisation>> {
    public:
        AutoRegressive(
            NamedShapelyParameters& shapely_parameters,
            std::string name = "AutoRegressive"
        ):
            torch::nn::Module(name),
            coefficients(ShapelyModule::template register_next_shapely_parameter<CoefficientsParameterisation>(shapely_parameters))
        { }

        void shapely_reset(void) override {
            if (enabled()) coefficients = ShapelyModule::template register_shapely_parameter<CoefficientsParameterisation>(coefficients->name(), coefficients->shapely_parameter_clone());
        }

        torch::Tensor forward(torch::Tensor x) const {
            return AutoRegressive_forward(std::move(x), coefficients->get());
        }

        torch::Tensor barrier(torch::Tensor scaling) const {
            return scaling*coefficients->barrier().mean();
        }

        torch::OrderedDict<std::string, std::vector<std::vector<torch::indexing::TensorIndex>>> observations_by_parameter(
            const torch::Tensor& x,
            bool recursive = true
        ) {
            return AutoRegressive_observations_by_parameter(x, coefficients, recursive);
        }

        bool enabled(void) const {
            return coefficients->enabled();
        }

        auto get(void) const {
            return coefficients->get();
        }

    private:
        std::shared_ptr<CoefficientsParameterisation> coefficients;
};

#endif

