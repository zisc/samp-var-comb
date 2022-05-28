#ifndef PROBABILISTIC_LOGSUBEXP_HPP_GUARD
#define PROBABILISTIC_LOGSUBEXP_HPP_GUARD

#include <torch/torch.h>

// Calculates logsubexp(x,y) = log(exp(x) - exp(y)) in a numerically stable fashion.

torch::Tensor logsubexp(const torch::Tensor& x, const torch::Tensor& y);
torch::Tensor logsubexp(const torch::Tensor& x, const torch::Scalar& y);
torch::Tensor logsubexp(const torch::Scalar& x, const torch::Tensor& y);
torch::Tensor logsubexp(const torch::Scalar& x, const torch::Scalar& y);

#endif

