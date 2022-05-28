#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <torch/torch.h>
#include <libtorch_support/logsubexp.hpp>
#include <libtorch_support/indexing.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/Mixture.hpp>
#include <modelling/distribution/Normal.hpp>
#include <seed_torch_rng.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace bdata = boost::unit_test::data;

class distribution_dataset {
    public:
        using sample = std::shared_ptr<Distribution>;
        enum { arity = 1 };

        class iterator {
            public:
                iterator():
                    N([]() {
                        auto mean = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble);
                        auto std_dev = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble).square();
                        return ManufactureNormal({{"X", mean}}, {{"X", std_dev}});
                    }()),
                    Mix([&]() {
                        auto mean = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble);
                        auto std_dev = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble).square();
                        std::shared_ptr<Distribution> N2 = ManufactureNormal({{"X", mean}}, {{"X", std_dev}});
                        auto weights = torch::empty({2}, torch::kDouble);
                        auto weights_a = weights.accessor<double,1>();
                        weights_a[0] = 0.6;
                        weights_a[1] = 0.4;
                        return ManufactureMixture({N, N2}, weights);
                    }()),
                    n(0)
                { }

                std::shared_ptr<Distribution> operator*() const {
                    switch (n) {
                        case 0:
                            return N;
                        case 1:
                            return Mix;
                        default:
                            throw std::logic_error("distribution_dataset::iterator::n not in {0,1}");
                    }
                }

                void operator++() { ++n; }

            private:
                std::shared_ptr<Distribution> N;
                std::shared_ptr<Distribution> Mix;
                int n;
        };

        distribution_dataset() { seed_torch_rng(); }

        bdata::size_t size() const { return 2; }

        iterator begin() const { return iterator(); }
};

namespace boost { namespace unit_test { namespace data { namespace monomorphic {
    template <>
    struct is_dataset<distribution_dataset> : boost::mpl::true_ {};
}}}}

BOOST_DATA_TEST_CASE(interval_test, distribution_dataset(), X) {
    seed_torch_rng();

    // auto mean = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble);
    // auto std_dev = torch::normal(0.0, 1.0, {10}, c10::nullopt, torch::kDouble).square();
    // auto X = ManufactureNormal({{"X", mean}}, {{"X", std_dev}});
    auto x1 = X->draw();
    auto x2 = X->draw();
    
    torch::OrderedDict<std::string, torch::Tensor> lb, ub;
    lb.insert("X", torch::min(x1["X"], x2["X"]));
    ub.insert("X", torch::max(x1["X"], x2["X"]));
    auto lbsc = lb["X"].index({0}).item<double>();
    auto ubsc = ub["X"].index({0}).item<double>();
    
    auto PrXinI = X->interval_probability(lb, ub)["X"];
    auto PrXinIc = X->interval_complement_probability(lb, ub)["X"];
    auto logPrXinI = X->log_interval_probability(lb, ub)["X"];
    auto logPrXinIc = X->log_interval_complement_probability(lb, ub)["X"];

    auto PrXinR = X->interval_complement_probability(ub, lb)["X"];
    auto PrXinRc = X->interval_probability(ub, lb)["X"];

    auto inf = std::numeric_limits<double>::infinity();
    auto PrXltub1 = X->cdf(ubsc)["X"];
    auto PrXltub2 = X->interval_probability(-inf, ubsc)["X"];
    auto PrXgtub1 = X->ccdf(ubsc)["X"];
    auto PrXgtub2 = X->interval_complement_probability(-inf, ubsc)["X"];
    auto PrXgtlb1 = X->ccdf(lbsc)["X"];
    auto PrXgtlb2 = X->interval_probability(lbsc, inf)["X"];
    auto PrXltlb1 = X->cdf(lbsc)["X"];
    auto PrXltlb2 = X->interval_complement_probability(lbsc, inf)["X"];
    auto logPrXltub1 = X->log_cdf(ubsc)["X"];
    auto logPrXltub2 = X->log_interval_probability(-inf, ubsc)["X"];
    auto logPrXgtub1 = X->log_ccdf(ubsc)["X"];
    auto logPrXgtub2 = X->log_interval_complement_probability(-inf, ubsc)["X"];
    auto logPrXgtlb1 = X->log_ccdf(lbsc)["X"];
    auto logPrXgtlb2 = X->log_interval_probability(lbsc, inf)["X"];
    auto logPrXltlb1 = X->log_cdf(lbsc)["X"];
    auto logPrXltlb2 = X->log_interval_complement_probability(lbsc, inf)["X"];

    // Uncomment this line and the corresponding test below when
    // missing::handle_na warns on infinite values instead of throwing.
    // auto logPrXinR = X->log_interval_complement_probability(ub, lb)["X"];

    // Pr(X in I) >= 0
    BOOST_TEST(static_cast<torch::Tensor>(PrXinI.ge(0.0).all()).item<bool>());

    // Pr(X in Ic) >= 0
    BOOST_TEST(static_cast<torch::Tensor>(PrXinIc.ge(0.0).all()).item<bool>());

    // Pr(X in I) + Pr(X in Ic) = 1
    BOOST_TEST(static_cast<torch::Tensor>((PrXinI + PrXinIc - 1.0).abs().sum().lt(1e-6)).item<bool>());

    // logPr(X in I) = logsubexp(0, logPr(X in Ic))
    BOOST_TEST(static_cast<torch::Tensor>((logPrXinI - logsubexp(0, logPrXinIc)).abs().sum().lt(1e-6)).item<bool>());

    // logPr(X in I) = log(Pr(X in I))
    BOOST_TEST(static_cast<torch::Tensor>((logPrXinI - PrXinI.log()).abs().sum().lt(1e-6)).item<bool>());

    // logPr(X in Ic) = log(Pr(X in Ic))
    BOOST_TEST(static_cast<torch::Tensor>((logPrXinIc - PrXinIc.log()).abs().sum().lt(1e-6)).item<bool>());

    // Pr(X in R) = 1.0
    BOOST_TEST(static_cast<torch::Tensor>((PrXinR - 1.0).abs().sum().lt(1e-6)).item<bool>());

    // Pr(X in Rc) = 0.0
    BOOST_TEST(static_cast<torch::Tensor>(PrXinRc.abs().sum().lt(1e-6)).item<bool>());

    // Uncomment this test when missing::handle_na warns on infinite values rather than throws.
    // logPr(X in R) = 0.0
    // BOOST_TEST(logPrXinR.abs().sum().lt(1e-6).item<bool>());

    BOOST_TEST(static_cast<torch::Tensor>((PrXltub1 - PrXltub2).abs().sum().lt(1e-6)).item<bool>());

    BOOST_TEST(static_cast<torch::Tensor>((PrXgtub1 - PrXgtub2).abs().sum().lt(1e-6)).item<bool>());

    BOOST_TEST(static_cast<torch::Tensor>((PrXgtlb1 - PrXgtlb2).abs().sum().lt(1e-6)).item<bool>());

    BOOST_TEST(static_cast<torch::Tensor>((PrXltlb1 - PrXltlb2).abs().sum().lt(1e-6)).item<bool>());

    BOOST_TEST(static_cast<torch::Tensor>((logPrXltub1 - logPrXltub2).abs().sum().lt(1e-6)).item<bool>());

    BOOST_TEST(static_cast<torch::Tensor>((logPrXgtub1 - logPrXgtub2).abs().sum().lt(1e-6)).item<bool>());

    BOOST_TEST(static_cast<torch::Tensor>((logPrXgtlb1 - logPrXgtlb2).abs().sum().lt(1e-6)).item<bool>());

    BOOST_TEST(static_cast<torch::Tensor>((logPrXltlb1 - logPrXltlb2).abs().sum().lt(1e-6)).item<bool>());
}

