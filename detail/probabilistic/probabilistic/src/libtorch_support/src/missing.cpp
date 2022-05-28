#include <torch/torch.h>
#include <libtorch_support/missing.hpp>

// See
// https://pytorch.org/tutorials/advanced/cpp_autograd.html#using-custom-autograd-function-in-c
// https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html
//
// A version of torch::where that doesn't corrupt gradients when infs, nans and missing values
// are present. Turns out the current implementation doesn't work, will revisit if required.

/*
using namespace torch::autograd;

namespace missing {

    class WhereImpl : public Function<WhereImpl> {
        public:
            static torch::Tensor forward(
                AutogradContext *ctx,
                const torch::Tensor& condition,
                const torch::Tensor& value_if_true,
                const torch::Tensor& value_if_false
            ) {
                ctx->save_for_backward({condition});
                return torch::where(condition, value_if_true, value_if_false);
            }

            static tensor_list backward(
                AutogradContext *ctx,
                tensor_list grad_outputs
            ) {
                auto saved = ctx->get_saved_variables();
                auto condition = saved[0];
                auto grad_output = grad_outputs[0];
                auto zeros = grad_output.new_zeros(grad_output.sizes());
                return {
                    zeros,
                    torch::where(condition, grad_output, zeros),
                    torch::where(condition, zeros, grad_output)
                };
            }
    };

    torch::Tensor where(const torch::Tensor& condition, const torch::Tensor& value_if_true, const torch::Tensor& value_if_false) {
        return WhereImpl::apply(condition, value_if_true, value_if_false);
    }

    torch::Tensor where(const torch::Tensor& condition, const torch::Tensor& value_if_true, const torch::Scalar& value_if_false) {
        return WhereImpl::apply(condition, value_if_true, torch::full({}, value_if_false, value_if_false.type()));
    }

    torch::Tensor where(const torch::Tensor& condition, const torch::Scalar& value_if_true, const torch::Tensor& value_if_false) {
        return WhereImpl::apply(condition, torch::full({}, value_if_true, value_if_true.type()), value_if_false);
    }

    torch::Tensor where(const torch::Tensor& condition, const torch::Scalar& value_if_true, const torch::Scalar& value_if_false) {
        return WhereImpl::apply(
            condition,
            torch::full({}, value_if_true, value_if_true.type()),
            torch::full({}, value_if_false, value_if_false.type())
        );
    }
}
*/

