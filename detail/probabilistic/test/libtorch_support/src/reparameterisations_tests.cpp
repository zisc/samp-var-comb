#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <torch/torch.h>
#include <libtorch_support/reparameterisations.hpp>

namespace bdata = boost::unit_test::data;
using namespace torch::indexing;

BOOST_DATA_TEST_CASE(
    reparameterisations_test,
    bdata::make({-10.0, -1.1, -0.5, 0.0, 1.0}) * bdata::make({2.0, 5.0, 13.0, 20.0}),
    lb,
    ub
) {
    double delta = ub - lb;
    auto draws = torch::normal(0.5*(ub + lb), delta, {10}, c10::nullopt, torch::kDouble);

    BOOST_TEST(static_cast<torch::Tensor>((draws - sigmoid_unbounded_above_inv(sigmoid_unbounded_above(draws, lb ,ub), lb, ub)).abs().max()).item<double>()/delta <= 0.1);
    BOOST_TEST((static_cast<torch::Tensor>((draws - sigmoid_unbounded_inv_approx(sigmoid_unbounded(draws, lb, ub), lb, ub)).abs().max()).item<double>()/delta) <= 2.0);
    BOOST_TEST(static_cast<torch::Tensor>((draws - sigmoid_unbounded_inv(sigmoid_unbounded(draws, lb, ub), lb, ub)).abs().max()).item<double>()/delta <= 0.001);
}

