#include <utility>
#include <Faddeeva.hh>
#include <torch/torch.h>
#include <libtorch_support/standard_normal_log_cdf.hpp>
#include <cmath>

// See
// http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package
// https://pytorch.org/tutorials/advanced/cpp_autograd.html#using-custom-autograd-function-in-c
// https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html

using namespace torch::autograd;

constexpr double inv_sqrt_2 = 0.7071067811865475244008443621048490392848359376884740365883398689;
constexpr double log_2 = 0.6931471805599453094172321214581765680755001343602552541206800094;
constexpr double sqrt_twoonpi = 0.7978845608028653558798921198687637369517172623298693153318516593;

/*
class ZLogCDFImpl : public Function<ZLogCDFImpl> {
    public:
        static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input) {
            auto output = input.clone().toType(torch::kDouble);
            auto *output_data_ptr = output.data_ptr<double>();
            auto output_size = output.numel();
            for (decltype(output_size) i = 0; i != output_size; ++i) {
                double z = output_data_ptr[i];
                if (z <= 0.0) {
                    output_data_ptr[i] = std::log(Faddeeva::erfcx(-inv_sqrt_2*z)) - 0.5*z*z - log_2;
                } else {
                    output_data_ptr[i] = std::log1p(-0.5*Faddeeva::erfc(inv_sqrt_2*z));
                }
            }
            ctx->save_for_backward({input});
            return output;
        }

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto grad_output = grad_outputs[0];

            auto doutdin = input.clone().toType(torch::kDouble);
            auto *doutdin_ptr = doutdin.data_ptr<double>();
            auto doutdin_size = doutdin.numel();
            for (decltype(doutdin_size) i = 0; i != doutdin_size; ++i) {
                double z = doutdin_ptr[i];
                if (z <= 0.0) {
                    doutdin_ptr[i] = sqrt_twoonpi/Faddeeva::erfcx(-inv_sqrt_2*z);
                } else {
                    doutdin_ptr[i] = sqrt_twoonpi*std::exp(-0.5*z*z)/(Faddeeva::erf(inv_sqrt_2*z) + 1.0);
                }
            }

            return {grad_output*doutdin};
        }
};

torch::Tensor standard_normal_log_cdf(torch::Tensor x) { return ZLogCDFImpl::apply(std::move(x)); }
*/

torch::Tensor standard_normal_log_cdf(torch::Tensor x) {
    auto le_0 = x.le(0.0);
    auto gt_0 = le_0.logical_not();
    auto x_le_0 = x.index({le_0});
    auto out = x.new_empty(x.sizes());
    out.index_put_({le_0}, torch::log(torch::special::erfcx(-inv_sqrt_2*x_le_0)) - 0.5*x_le_0.square() - log_2);
    out.index_put_({gt_0}, torch::log1p(-0.5*torch::special::erfc(inv_sqrt_2*x.index({gt_0}))));
    return out;
}

