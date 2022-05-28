#include <boost/test/unit_test.hpp>
#include <torch/torch.h>
#include <seed_torch_rng.hpp>

BOOST_AUTO_TEST_CASE(create_tensor) {
    seed_torch_rng();
    auto Z = torch::normal(0.0, 1.0, {1}, c10::nullopt, torch::kDouble);
    BOOST_TEST(Z.item<double>() < 8.0);
}

