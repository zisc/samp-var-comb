#include <utility>
#include <Faddeeva.hh>
#include <torch/torch.h>
#include <libtorch_support/erfcx.hpp>

// See
// http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package
// https://pytorch.org/tutorials/advanced/cpp_autograd.html#using-custom-autograd-function-in-c
// https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html

using namespace torch::autograd;

constexpr double twoonrootpi = 1.1283791670955125738961589031215451716881012586579977136881714434;

class ErfcxImpl : public Function<ErfcxImpl> {
    public:
        static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input) {
            auto output = input.clone().toType(torch::kDouble);
            auto *output_data_ptr = output.data_ptr<double>();
            auto output_size = output.numel();
            for (decltype(output_size) i = 0; i != output_size; ++i) {
                output_data_ptr[i] = Faddeeva::erfcx(output_data_ptr[i]);
            }
            ctx->save_for_backward({std::move(input), output});
            return output;
        }

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto output = saved[1];
            auto grad_output = grad_outputs[0];
            return {grad_output*(2*input*output - twoonrootpi)};
        }
};

torch::Tensor erfcx(torch::Tensor x) { return ErfcxImpl::apply(std::move(x)); }

