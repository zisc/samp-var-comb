#include <boost/test/unit_test.hpp>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <libtorch_support/logsubexp.hpp>
#include <modelling/distribution/Normal.hpp>
#include <modelling/distribution/Mixture.hpp>
#include <seed_torch_rng.hpp>
#include <cmath>
#include <memory>

BOOST_AUTO_TEST_CASE(mixture_test) {
    seed_torch_rng();

    auto mean_1 = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble);
    auto std_dev_1 = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble).square();
    std::shared_ptr<Distribution> X1 = ManufactureNormal({{"X", mean_1}}, {{"X", std_dev_1}});

    auto mean_2 = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble);
    auto std_dev_2 = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble).square();
    std::shared_ptr<Distribution> X2 = ManufactureNormal({{"X", mean_2}}, {{"X", std_dev_2}});

    auto X = ManufactureMixture({X1, X2}, torch::full({2}, 0.5, torch::kDouble));
    auto x = X->draw();

    auto X_cdf_x = X->cdf(x)[0].value();
    // std::cout << X_cdf_x << " = X_cdf_x\n";

    auto X_logcdf_x = X->log_cdf(x)[0].value();
    // std::cout << X_logcdf_x << " = X_logcdf_x\n";

    auto X_ccdf_x = X->ccdf(x)[0].value();
    // std::cout << X_ccdf_x << " = X_ccdf_x\n";

    auto X_logccdf_x = X->log_ccdf(x)[0].value();
    // std::cout << X_logccdf_x << " = X_logccdf_x\n";

    // Pr(X < x) + Pr(X > x) = 1
    BOOST_TEST(static_cast<torch::Tensor>(X_cdf_x + X_ccdf_x - 1.0).abs().sum().lt(1e-6).item<bool>());

    // logPr(X > X) = logsubexp(0, logPr(X < x))
    BOOST_TEST(static_cast<torch::Tensor>(X_logccdf_x - logsubexp(0.0, X_logcdf_x)).abs().sum().lt(1e-6).item<bool>());

    // logPr(X < x) = log(Pr(X < x))
    BOOST_TEST(static_cast<torch::Tensor>(X_logcdf_x - X_cdf_x.log()).abs().sum().lt(1e-6).item<bool>());

    // logPr(X > x) = log(Pr(X > x))
    BOOST_TEST(static_cast<torch::Tensor>(X_logccdf_x - X_ccdf_x.log()).abs().sum().lt(1e-6).item<bool>());
}

