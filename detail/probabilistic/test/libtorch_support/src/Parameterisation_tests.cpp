#include <boost/test/unit_test.hpp>
#include <torch/torch.h>
#include <libtorch_support/Parameterisation.hpp>
#include <seed_torch_rng.hpp>
#include <limits>

BOOST_AUTO_TEST_CASE(simplex) {
    seed_torch_rng();
    auto simplex_draw = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble).softmax(0);
    Simplex simplex_parameterisation(
        torch::full({1}, true, torch::kBool),
        "simplex",
        simplex_draw.clone(),
        torch::full({1}, 0.8, torch::kDouble),
        torch::full({1}, std::numeric_limits<double>::quiet_NaN(), torch::kDouble)
    );

    auto simplex_parameterisation_get = simplex_parameterisation.get();

    BOOST_TEST(static_cast<torch::Tensor>((simplex_draw - simplex_parameterisation_get).abs().sum().lt(1e-6)).item<bool>());
    BOOST_TEST(static_cast<torch::Tensor>((simplex_parameterisation_get.sum() - 1.0).abs().lt(1e-6)).item<bool>());
}

