#include <utility>
#include <torch/torch.h>
#include <libtorch_support/logsubexp.hpp>

// See
// https://pytorch.org/tutorials/advanced/cpp_autograd.html#using-custom-autograd-function-in-c
// https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html
//
// Calculates logsubexp(x,y) = log(exp(x) - exp(y)) and derivatives in a numerically stable fashion.
// The key risk to avoid is overflow or underflow upon computing the exponential terms.
// Note that logsubexp(x,y) = z + log(exp(x-z) - exp(y-z)) for all z. This implementations sets
// z = max(x,y) to mitigate this risk for (x,y) pairs that are not extremely far apart. 

using namespace torch::autograd;

class LogsubexpImpl : public Function<LogsubexpImpl> {
    public:
        static torch::Tensor forward(AutogradContext *ctx, const torch::Tensor& x, const torch::Tensor& y) {
            auto max_x_y = torch::max(x, y);
            auto exp_x_minus_max = (x - max_x_y).exp();
            auto exp_y_minus_max = (y - max_x_y).exp();
            auto output = max_x_y + (exp_x_minus_max - exp_y_minus_max).log();
            ctx->save_for_backward({std::move(exp_x_minus_max), std::move(exp_y_minus_max)});
            return output;
        }

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            auto saved = ctx->get_saved_variables();
            auto exp_x_minus_max = saved[0];
            auto exp_y_minus_max = saved[1];
            auto grad_output = grad_outputs[0];
            auto denominator = exp_x_minus_max - exp_y_minus_max;
            auto grad_x = exp_x_minus_max/denominator;
            auto grad_y = -exp_y_minus_max/denominator;
            return {grad_output*grad_x, grad_output*grad_y};
        }
};

torch::Tensor logsubexp(const torch::Tensor& x, const torch::Tensor& y) {
    return LogsubexpImpl::apply(x, y);
}

torch::Tensor logsubexp(const torch::Tensor& x, const torch::Scalar& y) {
    return logsubexp(x, torch::full({}, y, y.type()));
}

torch::Tensor logsubexp(const torch::Scalar& x, const torch::Tensor& y) {
    return logsubexp(torch::full({}, x, x.type()), y);
}

torch::Tensor logsubexp(const torch::Scalar& x, const torch::Scalar& y) {
    return logsubexp(torch::full({}, x, x.type()), torch::full({}, y, y.type()));
}

