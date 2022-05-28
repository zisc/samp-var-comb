#ifndef PROBABILISTIC_LIBTORCH_SUPPORT_DERIVATIVES_HPP_GUARD
#define PROBABILISTIC_LIBTORCH_SUPPORT_DERIVATIVES_HPP_GUARD

#include <utility>
#include <std_specialisations/hash.hpp>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>

enum class JacobianMode { Auto, Forward, Reverse };

// Output of jacobian is dy/dx as a tensor of shape cat(y.sizes(), x.sizes()).
torch::Tensor jacobian(
    const torch::Tensor& y,
    const torch::Tensor& x,
    JacobianMode mode = JacobianMode::Auto,
    bool create_graph = false,
    bool allow_unused = false,
    double na_jac = missing::na
);

template<class T>
torch::OrderedDict<T, torch::Tensor> jacobian(
    const torch::OrderedDict<T, torch::Tensor>& y,
    const torch::Tensor& x,
    JacobianMode mode = JacobianMode::Auto,
    bool create_graph = false,
    bool allow_unused = false,
    double na_jac = missing::na
) {
    torch::OrderedDict<T, torch::Tensor> out;
    for (const auto& y_item : y) {
        out.insert(y_item.key(), jacobian(y_item.value(), x, mode, create_graph, allow_unused, na_jac));
    }
    return out;
}

template<class T>
torch::OrderedDict<T, torch::Tensor> jacobian(
    const torch::Tensor& y,
    const torch::OrderedDict<T, torch::Tensor>& x,
    JacobianMode mode = JacobianMode::Auto,
    bool create_graph = false,
    bool allow_unused = false,
    double na_jac = missing::na
) {
    torch::OrderedDict<T, torch::Tensor> out;
    for (const auto& x_item : x) {
        out.insert(x_item.key(), jacobian(y, x_item.value(), mode, create_graph, allow_unused, na_jac));
    }
    return out;
}

template<class T1, class T2>
torch::OrderedDict<T1, torch::OrderedDict<T2, torch::Tensor>> jacobian(
    const torch::OrderedDict<T1, torch::Tensor>& y,
    const torch::OrderedDict<T2, torch::Tensor>& x,
    JacobianMode mode = JacobianMode::Auto,
    bool create_graph = false,
    bool allow_unused = false,
    double na_jac = missing::na
) {
    torch::OrderedDict<T1, torch::OrderedDict<T2, torch::Tensor>> out;
    out.reserve(y.size());
    for (const auto& y_item : y) {
        torch::OrderedDict<T2, torch::Tensor> out_nested;
        out_nested.reserve(x.size());
        const auto& y_item_value = y_item.value();
        for (const auto& x_item : x) {
            out_nested.insert(x_item.key(), jacobian(y_item_value, x_item.value(), mode, create_graph, allow_unused, na_jac));
        }
        out.insert(y_item.key(), std::move(out_nested));
    }
    return out;
}

// For torch::Tensors y and x, output of hessian is d2y/dx2 as a tensor of shape cat(y.sizes(), x.sizes(), x.sizes()).
template<class Y, class X>
auto hessian(
    const Y& y,
    const X& x,
    JacobianMode mode_nested = JacobianMode::Auto,
    JacobianMode mode_enclosing = JacobianMode::Auto,
    bool create_graph = false,
    bool allow_unused = false,
    double na_hess = missing::na
) {
    return jacobian(jacobian(y, x, mode_nested, true, allow_unused, na_hess), x, mode_enclosing, create_graph, allow_unused, na_hess);
}

#endif

