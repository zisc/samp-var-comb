#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <torch/torch.h>
#include <modelling/model/ProbabilisticModule.hpp>
#include <modelling/model/ARARCHTX.hpp>
#include <seed_torch_rng.hpp>
#include <limits>
#include <stdexcept>

namespace bdata = boost::unit_test::data;

class module_dataset {
    public:
        using sample = std::shared_ptr<ProbabilisticModule>;
        enum { arity = 1 };

        class iterator {
            public:
                iterator():
                    ararch([]() {
                        ShapelyParameter null_param;
                        
                        ShapelyParameter mu = {torch::full({1}, 2.0, torch::kDouble)};
                        
                        ShapelyParameter ar = {torch::full({3}, std::numeric_limits<double>::quiet_NaN(), torch::kDouble)};
                        auto ar_a = ar.parameter.accessor<double,1>();
                        ar_a[0] = 0.3; ar_a[1] = 0.2; ar_a[2] = 0.1;

                        ShapelyParameter sigma2 = {torch::full({1}, 1.0, torch::kDouble)};

                        ShapelyParameter arch = {torch::full({2}, std::numeric_limits<double>::quiet_NaN(), torch::kDouble)};
                        auto arch_a = arch.parameter.accessor<double,1>();
                        ar_a[0] = 0.2; ar_a[1] = 0.1;
                        
                        NamedShapelyParameters sp = {{
                            {"mu", mu},
                            {"mean_exogenous_coef", null_param},
                            {"ar", ar},
                            {"sigma2", sigma2},
                            {"var_exogenous_coef", null_param},
                            {"arch", arch}
                        }};

                        Buffers b = {{
                            torch::full({}, 0.0, torch::kDouble),   // var_transformation_crimp
                            torch::full({}, 1.0, torch::kDouble),   // var_transformation_catch
                            torch::full({1}, 'X', torch::kChar)
                        }};

                        return ManufactureARARCHTX(sp, b);
                    }())
                { }
                            

                std::shared_ptr<ProbabilisticModule> operator*() const {
                    switch (n) {
                        case 0:
                            return ararch;
                        default:
                            throw std::logic_error("module_dataset::iterator::n != 0");
                    }
                }

                void operator++() { ++n; }

            private:
                std::shared_ptr<ProbabilisticModule> ararch;
                int n;
        };

        module_dataset() { seed_torch_rng(); }

        bdata::size_t size() const { return 1; }

        iterator begin() const { return iterator(); }
};

namespace boost { namespace unit_test { namespace data { namespace monomorphic {
    template <>
    struct is_dataset<module_dataset> : boost::mpl::true_ {};
}}}}

/*
BOOST_DATA_TEST_CASE(generate_test, module_dataset(), model) {
    seed_torch_rng();
    BOOST_TEST(true);
    // TODO: Test that ARARCHTX generates from the same model that it forecasts.
}
*/

