#include <boost/test/unit_test.hpp>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <libtorch_support/logsubexp.hpp>
#include <seed_torch_rng.hpp>
#include <cmath>

BOOST_AUTO_TEST_CASE(logsubexp_test) {
    seed_torch_rng();
    
    // Tensor, Tensor
    
    auto y = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::requires_grad().dtype(torch::kDouble));
    auto x = y + torch::rand({10}, torch::requires_grad().dtype(torch::kDouble));
    
    auto logsubexp_xy_unstable = (x.exp() - y.exp()).log();
    auto logsubexp_xy_stable = logsubexp(x, y);

    BOOST_TEST(static_cast<torch::Tensor>(logsubexp_xy_stable - logsubexp_xy_unstable).abs().sum().lt(1e-6).item<bool>());

    auto grad_x_unstable = jacobian(logsubexp_xy_unstable, x).diag();
    auto grad_y_unstable = jacobian(logsubexp_xy_unstable, y).diag();

    auto grad_x_stable = jacobian(logsubexp_xy_stable, x).diag();
    auto grad_y_stable = jacobian(logsubexp_xy_stable, y).diag();

    BOOST_TEST(static_cast<torch::Tensor>(grad_x_stable - grad_x_unstable).abs().sum().lt(1e-6).item<bool>());
    BOOST_TEST(static_cast<torch::Tensor>(grad_y_stable - grad_y_unstable).abs().sum().lt(1e-6).item<bool>());

    // Tensor, Scalar

    auto min_y = y.min().item<double>();

    auto logsubexp_x_min_y_unstable = (x.exp() - std::exp(min_y)).log();
    auto logsubexp_x_min_y_stable = logsubexp(x, min_y);

    BOOST_TEST(static_cast<torch::Tensor>(logsubexp_x_min_y_stable - logsubexp_x_min_y_unstable).abs().sum().lt(1e-6).item<bool>());

    grad_x_unstable = jacobian(logsubexp_x_min_y_unstable, x).diag();
    grad_x_stable = jacobian(logsubexp_x_min_y_stable, x).diag();

    BOOST_TEST(static_cast<torch::Tensor>(grad_x_stable - grad_x_unstable).abs().sum().lt(1e-6).item<bool>());

    // Scalar, Tensor

    auto max_x = x.max().item<double>();

    auto logsubexp_max_x_y_unstable = (std::exp(max_x) - y.exp()).log();
    auto logsubexp_max_x_y_stable = logsubexp(max_x, y);

    BOOST_TEST(static_cast<torch::Tensor>(logsubexp_max_x_y_stable - logsubexp_max_x_y_unstable).abs().sum().lt(1e-6).item<bool>());

    grad_y_stable = jacobian(logsubexp_max_x_y_stable, y).diag();
    grad_y_unstable = jacobian(logsubexp_max_x_y_unstable, y).diag();


    BOOST_TEST(static_cast<torch::Tensor>(grad_y_stable - grad_y_unstable).abs().sum().lt(1e-6).item<bool>());

    // Scalar, Scalar
    //
    auto logsubexp_max_x_min_y_unstable = std::log(std::exp(max_x) - std::exp(min_y));
    auto logsubexp_max_x_min_y_stable = logsubexp(max_x, min_y);

    BOOST_TEST(static_cast<torch::Tensor>(logsubexp_max_x_min_y_stable - logsubexp_max_x_min_y_unstable).abs().sum().lt(1e-6).item<bool>());
}

