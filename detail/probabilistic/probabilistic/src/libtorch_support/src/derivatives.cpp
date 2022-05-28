#include <limits>
#include <vector>
#include <torch/torch.h>
#include <libtorch_support/derivatives.hpp>

// Inspired by https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7.

torch::Tensor jacobian_reverse_mode(
    const torch::Tensor& y,
    const torch::Tensor& x,
    bool create_graph,
    bool allow_unused
) {
    auto x_shape = x.sizes();
    auto y_shape = y.sizes();
    auto flat_y = y.reshape({-1});
    auto ny = flat_y.numel();
    
    std::vector<int64_t> grad_shape;
    grad_shape.insert(grad_shape.end(), y_shape.cbegin(), y_shape.cend());
    grad_shape.insert(grad_shape.end(), x_shape.cbegin(), x_shape.cend());
    
    auto grad_output = torch::zeros_like(flat_y, torch::kDouble);
    auto grad_output_a = grad_output.accessor<double, 1>();
    
    std::vector<torch::Tensor> grad_vec;

    // If y and x don't belong on the same computation graph, then torch::autograd::grad returns an empty Tensor.
    // In fact, we want to return all zeros instead, so for the first grad_output, return zeroes if the gradient
    // is an empty vector.
    grad_output_a[0] = 1.0;
    auto grad_vec_0 = torch::autograd::grad({flat_y}, {x}, {grad_output}, true, create_graph, allow_unused).at(0);
    if (allow_unused && grad_vec_0.numel() == 0) {
        return y.new_zeros(grad_shape, torch::kDouble);
    }
    grad_vec.emplace_back(grad_vec_0.reshape(x_shape));
    grad_output_a[0] = 0.0;

    for (int64_t i = 1; i < ny; ++i) { // Reverse mode means this loop is over y. Note that i = 0 was completed above.
        grad_output_a[i] = 1.0;
        grad_vec.emplace_back(torch::autograd::grad({flat_y}, {x}, {grad_output}, true, create_graph, allow_unused).at(0).reshape(x_shape));
        grad_output_a[i] = 0.0;
    }

    return torch::stack(grad_vec).reshape(grad_shape);
}


// Forward mode version inspired by https://colab.research.google.com/drive/1tcm7Lvdv0krpPdaYHtWe7NA2bDbEQ0uj
torch::Tensor jacobian_forward_mode(
    const torch::Tensor& y,
    const torch::Tensor& x,
    bool create_graph,
    bool allow_unused
) {
    auto x_shape = x.sizes();
    auto flat_x = x.reshape({-1});
    auto nx = flat_x.numel();
    auto y_shape = y.sizes();
    auto flat_y = y.reshape({-1});
    auto ny = flat_y.numel();

    std::vector<int64_t> grad_shape;
    grad_shape.insert(grad_shape.end(), y_shape.cbegin(), y_shape.cend());
    grad_shape.insert(grad_shape.end(), x_shape.cbegin(), x_shape.cend());

    // To achieve forward mode with two reverse mode passes, we take the
    // derivative of the first reverse mode pass using any vector.
    auto any_vector = torch::ones({ny}, torch::requires_grad().dtype(torch::kDouble));
    auto vjp = torch::autograd::grad({flat_y}, {x}, {any_vector}, true, true, allow_unused).at(0)/*.reshape({-1})*/;
    if (allow_unused && vjp.numel() == 0) {
        return y.new_zeros(grad_shape, torch::kDouble);
    }
    vjp = vjp.reshape({-1});
    
    // Now pick off the vectors dy_./dx_i.
    auto grad_output = torch::zeros_like(flat_x, torch::kDouble);
    auto grad_output_a = grad_output.accessor<double, 1>();
    std::vector<torch::Tensor> grad_vec;
    for (int64_t i = 0; i != nx; ++i) { // Forward mode means this loop is over x.
        grad_output_a[i] = 1.0;
        grad_vec.emplace_back(torch::autograd::grad({vjp}, {any_vector}, {grad_output}, true, create_graph, allow_unused).at(0).reshape(y_shape));
        grad_output_a[i] = 0.0;
    }
    return torch::stack(grad_vec, y.ndimension()).reshape(grad_shape);
}

torch::Tensor jacobian(
    const torch::Tensor& y,
    const torch::Tensor& x,
    JacobianMode mode,
    bool create_graph,
    bool allow_unused,
    double na_jac
) {
    auto get_jac_no_missings = [&](const torch::Tensor& xn, const torch::Tensor& yn) {
        switch(mode) {
            case JacobianMode::Auto :
                if (yn.numel() > xn.numel()) {
                    return jacobian_forward_mode(yn, xn, create_graph, allow_unused);
                } else {
                    return jacobian_reverse_mode(yn, xn, create_graph, allow_unused);
                }
                break;
            case JacobianMode::Forward :
                return jacobian_forward_mode(yn, xn, create_graph, allow_unused);
                break;
            case JacobianMode::Reverse :
                return jacobian_reverse_mode(yn, xn, create_graph, allow_unused);
                break;
        }
    };

    if (na_jac == 0.0) {
        return get_jac_no_missings(x,y);
    } else {
        auto y_detach = y.detach();
        auto y_sizes = y_detach.sizes();

        auto x_detach = x.detach();
        auto x_sizes = x_detach.sizes();
        
        auto jac_ndimension = y_detach.ndimension() + x_detach.ndimension();
        std::vector<int64_t> jac_sizes; jac_sizes.reserve(jac_ndimension);
        for (const auto& yi : y_sizes) { jac_sizes.emplace_back(yi); }
        for (const auto& xi : x_sizes) { jac_sizes.emplace_back(xi); }

        auto x_is_missing = x_detach.eq(na_jac);
        while(x_is_missing.ndimension() < jac_ndimension) {
            x_is_missing.unsqueeze_(0);
        }
        x_is_missing = x_is_missing.expand(jac_sizes);

        auto y_is_missing = y_detach.eq(na_jac);
        while(y_is_missing.ndimension() < jac_ndimension) {
            y_is_missing.unsqueeze_(y_is_missing.ndimension());
        }
        y_is_missing = y_is_missing.expand(jac_sizes);

        auto jac_is_missing = torch::logical_or(x_is_missing, y_is_missing);

        auto jac = get_jac_no_missings(x,y);

        auto *jac_is_missing_ptr = jac_is_missing.data_ptr<bool>();
        auto *jac_ptr = jac.data_ptr<double>();
        auto numel = jac.numel();
        for (decltype(numel) i = 0; i != numel; ++i) {
            if (jac_is_missing_ptr[i]) {
                jac_ptr[i] = missing::na;
            }
        }

        return jac;
    }
}

