#ifndef PROBABILISTIC_LIBTORCH_SUPPORT_REPARAMETERISATIONS_HPP_GUARD
#define PROBABILISTIC_LIBTORCH_SUPPORT_REPARAMETERISATIONS_HPP_GUARD

#include <limits>
#include <torch/torch.h>

void enforce_parameter_space(torch::Tensor& x, double lb = -std::numeric_limits<double>::infinity(), double ub = std::numeric_limits<double>::infinity(), double scaling = 1.0);

double sigmoid_unbounded_above(double x);
double sigmoid_unbounded_above_inv(double y);
torch::Tensor sigmoid_unbounded_above(const torch::Tensor& x, double lb, double soft_ub, double scaling = 1.0);
torch::Tensor sigmoid_unbounded_above_inv(const torch::Tensor& y, double lb, double soft_ub, double scaling = 1.0);

double sigmoid_unbounded(double x, double soft_lb, double soft_ub, double scaling = 1.0);
double sigmoid_unbounded_inv(double y, double soft_lb, double soft_ub, double scaling = 1.0);
torch::Tensor sigmoid_unbounded(const torch::Tensor& x, double soft_lb, double soft_ub, double scaling = 1.0);
torch::Tensor sigmoid_unbounded_inv(const torch::Tensor& y, double soft_lb, double soft_ub, double scaling = 1.0);
torch::Tensor sigmoid_unbounded_inv_approx(const torch::Tensor& y, double soft_lb, double soft_ub);

torch::Tensor clamped_linear(torch::Tensor& x, double lb, double ub, double scaling = 1.0);
torch::Tensor clamped_linear(torch::Tensor& x, double lb, torch::Tensor ub, double scaling = 1.0);
torch::Tensor clamped_linear(torch::Tensor& x, torch::Tensor lb, double ub, double scaling = 1.0);
torch::Tensor clamped_linear(torch::Tensor& x, torch::Tensor lb, torch::Tensor ub, double scaling = 1.0);
torch::Tensor clamped_linear_inv(const torch::Tensor& y, double scaling = 1.0);

torch::Tensor clamped_sigmoid_unbounded_above(torch::Tensor& x, double lb, double soft_ub, double scaling = 1.0);
torch::Tensor clamped_sigmoid_unbounded_above_inv(const torch::Tensor&y, double lb, double soft_ub, double scaling = 1.0);
void clamped_sigmoid_unbounded_above_enforce_parameter_space(torch::Tensor& x);

#endif

