#include <boost/test/unit_test.hpp>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <libtorch_support/standard_normal_log_cdf.hpp>
#include <seed_torch_rng.hpp>

#include <iostream>

constexpr double n_sqrt2_inv = -0.7071067811865475244008443621048490392848359376884740365883398689;

BOOST_AUTO_TEST_CASE(standard_normal_log_cdf_test) {
    seed_torch_rng();
    auto draw = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::requires_grad().dtype(torch::kDouble));
    auto log_cdf_unstable = torch::log(0.5*torch::erfc(n_sqrt2_inv*draw));
    auto log_cdf_stable = standard_normal_log_cdf(draw);
    auto grad_log_cdf_unstable = jacobian(log_cdf_unstable, draw).diag();
    auto grad_log_cdf_stable = jacobian(log_cdf_stable, draw).diag();

    BOOST_TEST(static_cast<torch::Tensor>((log_cdf_stable - log_cdf_unstable).abs().sum().lt(1e-6)).item<bool>());
    BOOST_TEST(static_cast<torch::Tensor>((grad_log_cdf_stable - grad_log_cdf_unstable).abs().sum().lt(1e-6)).item<bool>());
}

