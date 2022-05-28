#include <boost/test/unit_test.hpp>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <libtorch_support/erfcx.hpp>
#include <seed_torch_rng.hpp>

BOOST_AUTO_TEST_CASE(erfcxs_test) {
    seed_torch_rng();
    auto draw = torch::normal(0.0, 0.5, {10}, c10::nullopt, torch::requires_grad().dtype(torch::kDouble));
    auto erfcx_draw_unstable = draw.square().exp() * draw.erfc();
    auto erfcx_draw_stable = erfcx(draw);
    auto grad_erfcx_draw_unstable = jacobian(erfcx_draw_unstable, draw).diag();
    auto grad_erfcx_draw_stable = jacobian(erfcx_draw_stable, draw).diag();

    BOOST_TEST(static_cast<torch::Tensor>(erfcx_draw_stable - erfcx_draw_unstable).abs().sum().lt(1e-6).item<bool>());
    BOOST_TEST(static_cast<torch::Tensor>(grad_erfcx_draw_stable - grad_erfcx_draw_unstable).abs().sum().lt(1e-6).item<bool>());
}

