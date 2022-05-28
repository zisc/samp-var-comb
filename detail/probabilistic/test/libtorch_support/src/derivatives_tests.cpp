#include <cmath>
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>
#include <iostream>

namespace bdata = boost::unit_test::data;

BOOST_DATA_TEST_CASE(
    jacobian_test,
    bdata::make({1,2,3,4})*bdata::make({1,2,3,4}),
    xnrows,
    xncols
) {
    torch::Tensor x = torch::rand({xnrows, xncols}, torch::requires_grad().dtype(torch::kDouble));
    auto x_sizes = x.sizes();
    auto x_a = x.accessor<double, 2>();

    auto ynrows = std::min(xnrows, xncols);
    auto u = x.expand({xnrows, -1, -1})
              .transpose(0,2)
              .transpose(0,1)
              .index({torch::indexing::Slice(0, ynrows), torch::indexing::Ellipsis});
    auto v = x.expand({xncols, -1, -1})
              .transpose(1,2)
              .transpose(0,1)
              .index({torch::indexing::Slice(0, ynrows), torch::indexing::Ellipsis});
    
    // y[i][j][k] = x[i][j]*x[k][i]
    torch::Tensor y = u*v;
    auto y_sizes = y.sizes();
    auto y_a = y.accessor<double ,3>();

    BOOST_TEST(y_sizes.size() == 3);
    BOOST_TEST(y_sizes.at(0) == ynrows);
    BOOST_TEST(y_sizes.at(1) == xncols);
    BOOST_TEST(y_sizes.at(2) == xnrows);

    for (int64_t i = 0; i != y_sizes[0]; ++i) {
        for (int64_t j = 0; j != y_sizes[1]; ++j) {
            for (int64_t k = 0; k != y_sizes[2]; ++k) {
                BOOST_TEST(std::abs(y_a[i][j][k] - x_a[i][j]*x_a[k][i]) < 1e-12);
                if (std::abs(y_a[i][j][k] - x_a[i][j]*x_a[k][i]) >= 1e-12) {
                    std::cout << "i = " << i << ", j = " << j << ", k = " << k << '\n';
                }
            }
        }
    }

    for (auto mode : {JacobianMode::Forward, JacobianMode::Reverse}) {
        torch::Tensor dydx = jacobian(y, x, mode);

        auto dydx_sizes = dydx.sizes();
        bool dydx_to_size = true;
        
        BOOST_TEST(dydx_sizes.size() == 5);
        if (dydx_sizes.size() != 5) { dydx_to_size = false; }

        for (int64_t i = 0; i < 3 && i < dydx_sizes.size(); ++i) {
            BOOST_TEST(dydx_sizes.at(i) == y_sizes.at(i));
            if (dydx_sizes.at(i) != y_sizes.at(i)) { dydx_to_size = false; }
        }

        for (int64_t i = 3; i < 5 && i < dydx_sizes.size(); ++i) {
            BOOST_TEST(dydx_sizes.at(i) == x_sizes.at(i-3));
            if (dydx_sizes.at(i) != x_sizes.at(i-3)) { dydx_to_size = false; }
        }

        if (dydx_to_size) {
            auto dydx_a = dydx.accessor<double, 5>();
            auto deriv_test = [&](
                int64_t di, int64_t dj, int64_t dk, int64_t dl, int64_t dm,
                int64_t xi, int64_t xj,
                int64_t c = 1
            ) {
                BOOST_TEST(std::abs(dydx_a[di][dj][dk][dl][dm] - c*x_a[xi][xj]) < 1e-12);
                auto this_dydx = dydx_a[di][dj][dk][dl][dm];
                auto this_x = x_a[xi][xj];
                if (std::abs(this_dydx - c*this_x) >= 1e-12) {
                    std::cout << "Test failed for " << (mode == JacobianMode::Forward ? "forward" : "reverse") << " mode dydx index "
                                 "[" << di << ',' << dj << ',' << dk << ',' << dl << ',' << dm << "]\n"
                                 "Indices of x where dydx is close to cx are: \n"
                              << (this_dydx - c*x).abs().lt(1e-12).nonzero() << "\n\n";
                }
            };
            for (int64_t i = 0; i != dydx_sizes[0]; ++i) {
                for (int64_t j = 0; j != dydx_sizes[1]; ++j) {
                    for (int64_t k = 0; k != dydx_sizes[2]; ++k) {
                        for (int64_t l = 0; l != dydx_sizes[3]; ++l) {
                            for (int64_t m = 0; m != dydx_sizes[4]; ++m) {
                                if (i == j && i == k && i == l && i == m) {
                                    deriv_test(
                                        i, i, i, i, i,
                                        i, i,
                                        2
                                    );
                                } else if (i == l && j == m) {
                                    deriv_test(
                                        i, j, k, i, j,
                                        k, i
                                    );
                                } else if (i == m && k == l) {
                                    deriv_test(
                                        i, j, k, k, i,
                                        i, j
                                    );
                                } else {
                                    BOOST_TEST(std::abs(dydx_a[i][j][k][l][m]) < 1e-12);
                                    if (std::abs(dydx_a[i][j][k][l][m]) >= 1e-12) {
                                        std::cout << "Test failed for " << (mode == JacobianMode::Forward ? "forward" : "reverse") << " mode dydx index "
                                                     "[" << i << ',' << j << ',' << k << ',' << l << ',' << m << "]\n";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }

}

