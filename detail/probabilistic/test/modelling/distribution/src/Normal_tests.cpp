#include <boost/test/unit_test.hpp>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <libtorch_support/logsubexp.hpp>
#include <modelling/distribution/Normal.hpp>
#include <seed_torch_rng.hpp>
#include <cmath>

BOOST_AUTO_TEST_CASE(normal_test) {
    seed_torch_rng();

    auto mean = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble);
    auto std_dev = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble).square();
    auto X = ManufactureNormal({{"X", mean}}, {{"X", std_dev}});
    auto x = X->draw();
    auto X_cdf_x = X->cdf(x)[0].value();
    auto X_logcdf_x = X->log_cdf(x)[0].value();
    auto X_ccdf_x = X->ccdf(x)[0].value();
    auto X_logccdf_x = X->log_ccdf(x)[0].value();

    // Pr(X < x) + Pr(X > x) = 1
    BOOST_TEST(static_cast<torch::Tensor>(X_cdf_x + X_ccdf_x - 1.0).abs().sum().lt(1e-6).item<bool>());

    // logPr(X > X) = logsubexp(0, logPr(X < x))
    BOOST_TEST(static_cast<torch::Tensor>(X_logccdf_x - logsubexp(0.0, X_logcdf_x)).abs().sum().lt(1e-6).item<bool>());

    // logPr(X < x) = log(Pr(X < x))
    BOOST_TEST(static_cast<torch::Tensor>(X_logcdf_x - X_cdf_x.log()).abs().sum().lt(1e-6).item<bool>());

    // logPr(X > x) = log(Pr(X > x))
    BOOST_TEST(static_cast<torch::Tensor>(X_logccdf_x - X_ccdf_x.log()).abs().sum().lt(1e-6).item<bool>());
}

