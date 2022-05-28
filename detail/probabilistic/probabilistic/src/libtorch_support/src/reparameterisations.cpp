#include <cmath>
#include <sstream>
#include <stdexcept>
#include <torch/torch.h>
#include <libtorch_support/reparameterisations.hpp>

template<typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
T dt_nextafter(T x, T y) {
    if (x == y) {
        return x;
    } else if (x < y) {
        return x + 1;
    } else {
        return x - 1;
    }
}

template<typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
auto dt_nextafter(T x, T y) {
    if (x == y) {
        return x;
    } else {
        return nextafter(x,y);
    }
}

template<class Condition, class Scalar>
auto get_minimum_value_to_satisfy_condition(
    Condition&& condition,
    Scalar value_first_guess
) {
    auto scalar_min = std::numeric_limits<Scalar>::lowest();
    auto scalar_max = std::numeric_limits<Scalar>::max();
    auto zero = static_cast<Scalar>(0);
    auto two = static_cast<Scalar>(2);
    
    auto value = value_first_guess;
    auto value_lower_bound = std::numeric_limits<Scalar>::lowest();     // These initial values indicate no known bound.
    auto value_upper_bound = std::numeric_limits<Scalar>::max();
    
    auto go_lower = [&value, &value_lower_bound, &value_upper_bound, scalar_min, two, zero]() {
        if (value_lower_bound == scalar_min && value > dt_nextafter(scalar_min/two, zero)) {
            value_upper_bound = value;
            if (value <= zero) {
                value = dt_nextafter(two*value, scalar_min);
            } else {
                value = dt_nextafter(value/two, zero);
            }
        } else {
            value_upper_bound = value;
            value -= (value - value_lower_bound)/two;
            if (value == value_upper_bound) value = dt_nextafter(value, value_lower_bound);
        }
    };
    
    auto go_higher = [&value, &value_lower_bound, &value_upper_bound, scalar_max, two, zero]() {
        if (value_upper_bound == scalar_max && value < dt_nextafter(scalar_max/two, zero)) {
            value_lower_bound = value;
            if (value >= zero) {
                value = dt_nextafter(two*value, scalar_max);
            } else {
                value = dt_nextafter(value/two, zero);
            }
        } else {
            value_lower_bound = value;
            value += (value_upper_bound - value)/two;
            if (value == value_lower_bound) value = dt_nextafter(value, value_upper_bound);
        }
    };
    
    auto throw_if_bounds_nonsensical = [&value, &value_lower_bound, &value_upper_bound]() {
        if (value < value_lower_bound) {
            throw std::logic_error("get_minimum_value_to_satisfy_condition has value < value_lower_bound.");
        } else if (value > value_upper_bound) {
            throw std::logic_error("get_minimum_value_to_satisfy_condition has value > value_upper_bound.");
        }
    };
    
    std::vector<Scalar> history;
    auto throw_if_loop = [&value, &value_upper_bound, &value_lower_bound, &history] () {
        if (std::find(history.begin(), history.end(), value) != history.end()) {
            std::ostringstream ss;
            ss << "A loop was found in get_minimum_value_to_satisfy_condition!\n"
                  "value_lower_bound = " << value_lower_bound << "\n"
                  "value_upper_bound = " << value_upper_bound << "\n"
                  "history: \n";
            for (auto iter = history.begin(); iter != history.end(); ++iter) {
                ss << *iter << '\n';
            }
            ss << value << '\n';
            throw std::logic_error(ss.str());
        } else {
            history.push_back(value);
        }
    };
    
    while (true) {
        throw_if_loop();
        throw_if_bounds_nonsensical();
        if (condition(value)) {                                                     // Just right or too high.
            if (value == scalar_min || !condition(dt_nextafter(value, scalar_min))) {   // Just right.
                break;
            } else {                                                                // Too high, go lower.
                go_lower();
            }
        } else {                                                                    // Too low, go higher.
            go_higher();
        }
    }
    
    return value;
}

void enforce_parameter_space(torch::Tensor& x, double lb, double ub, double scaling) {
    auto lb_scaled = lb*scaling;
    auto ub_scaled = ub*scaling;
    auto x_ptr = x.data_ptr<double>();
    auto x_size = x.numel();
    for (decltype(x_size) i = 0; i != x_size; ++i) {
        if (x_ptr[i] < lb_scaled) {
            x_ptr[i] = lb_scaled;
        } else if (x_ptr[i] > ub_scaled) {
            x_ptr[i] = ub_scaled;
        }
    }
}

constexpr double onepln2 = 1.6931471805599453094172321214581765680755001343602552541206800094;

double sigmoid_unbounded_above(double x) {
    return log1p(log1p(exp(onepln2*x)));
}

torch::Tensor sigmoid_unbounded_above(const torch::Tensor& x) {
    auto zeroes = x.new_full(x.sizes(), 0.0);
    return (onepln2*x).logaddexp(zeroes).log1p();
}

double sigmoid_unbounded_above_inv(double y) {
    return logl(expm1l(expm1l(y)))/onepln2;
}

torch::Tensor sigmoid_unbounded_above_inv(const torch::Tensor& y) {
    if (static_cast<torch::Tensor>(y.le(0.0).any()).item<bool>()) {
        throw std::logic_error("y.le(0.0).any().item<bool>()");
    }

    auto x = y.new_empty(y.sizes());
    auto *x_ptr = x.data_ptr<double>();
    auto *y_ptr = y.data_ptr<double>();
    for (int64_t i = 0; i != y.numel(); ++i) {
        x_ptr[i] = sigmoid_unbounded_above_inv(y_ptr[i]);
    }

    return x;
}

torch::Tensor sigmoid_unbounded_above(const torch::Tensor& x, double lb, double soft_ub, double scaling) {
    double delta = soft_ub - lb;
    return lb + 2.0*delta*sigmoid_unbounded_above(3.0*x/(delta*scaling))/3.0;
}

torch::Tensor sigmoid_unbounded_above_inv(const torch::Tensor& y, double lb, double soft_ub, double scaling) {
    double delta = soft_ub - lb;
    return delta*scaling*sigmoid_unbounded_above_inv(3.0*(y-lb)/(2.0*delta))/3.0;
}

double sigmoid_unbounded(double x, double soft_lb, double soft_ub, double scaling) {
    double avg = 0.5*(soft_ub + soft_lb);
    double rad = 0.5*(soft_ub - soft_lb);
    double rad_scaling = rad*scaling;
    return avg + rad*(sigmoid_unbounded_above(x/rad_scaling) - sigmoid_unbounded_above(-x/rad_scaling));
}

torch::Tensor sigmoid_unbounded(const torch::Tensor& x, double soft_lb, double soft_ub, double scaling) {
    double avg = 0.5*(soft_ub + soft_lb);
    double rad = 0.5*(soft_ub - soft_lb);
    double rad_scaling = rad*scaling;
    return avg + rad*(sigmoid_unbounded_above(x/rad_scaling) - sigmoid_unbounded_above(-x/rad_scaling));
}

double sigmoid_unbounded_inv(double y, double soft_lb, double soft_ub, double scaling) {
    double avg = 0.5*(soft_ub + soft_lb);
    return get_minimum_value_to_satisfy_condition(
        [&](double x) { return sigmoid_unbounded(x, soft_lb, soft_ub) >= y; },
        avg
    )*scaling;
}

torch::Tensor sigmoid_unbounded_inv(const torch::Tensor& y, double soft_lb, double soft_ub, double scaling) {
    auto x = y.new_empty(y.sizes());
    auto *x_ptr = x.data_ptr<double>();
    auto *y_ptr = y.data_ptr<double>();
    for (int64_t i = 0; i != y.numel(); ++i) {
        x_ptr[i] = sigmoid_unbounded_inv(y_ptr[i], soft_lb, soft_ub, scaling);
    }
    return x;
}

torch::Tensor sigmoid_unbounded_inv_approx(const torch::Tensor& y, double soft_lb, double soft_ub) {
    double avg = 0.5*(soft_ub + soft_lb);
    double rad = 0.5*(soft_ub - soft_lb);

    torch::Tensor ret = y.clone();

    auto ret_lt_lb = ret.lt(soft_lb);
    auto ret_gt_ub = ret.gt(soft_ub);

    ret.index_put_({ret_lt_lb}, -rad*sigmoid_unbounded_above_inv(-(y.index({ret_lt_lb})-avg)/rad));
    ret.index_put_({ret_gt_ub}, rad*sigmoid_unbounded_above_inv((y.index({ret_gt_ub})-avg)/rad));

    return ret;
}

torch::Tensor clamped_linear(torch::Tensor& x, double lb, double ub, double scaling) {
    lb *= scaling;
    ub *= scaling;

    auto x_ptr = x.data_ptr<double>();
    for (int64_t i = 0; i != x.numel(); ++i) {
        if (x_ptr[i] < lb) {
            x_ptr[i] = lb;
        } else if (x_ptr[i] > ub) {
            x_ptr[i] = ub;
        }
    }

    return x/scaling;
}

torch::Tensor clamped_linear(torch::Tensor& x, double lb, torch::Tensor ub, double scaling) {
    lb *= scaling;
    ub *= scaling;

    auto numel = x.numel();
    if (numel != ub.numel()) {
        throw std::logic_error("clamped_linear: x.numel() != ub.numel()");
    }

    auto x_ptr = x.data_ptr<double>();
    auto ub_ptr = ub.data_ptr<double>();
    for (decltype(numel) i = 0; i != numel; ++i) {
        if (x_ptr[i] < lb) {
            x_ptr[i] = lb;
        } else if (x_ptr[i] > ub_ptr[i]) {
            x_ptr[i] = ub_ptr[i];
        }
    }
    
    return x/scaling;
}

torch::Tensor clamped_linear(torch::Tensor& x, torch::Tensor lb, double ub, double scaling) {
    lb *= scaling;
    ub *= scaling;

    auto numel = x.numel();
    if (numel != lb.numel()) {
        throw std::logic_error("clamped_linear: x.numel() != ub.numel()");
    }

    auto x_ptr = x.data_ptr<double>();
    auto lb_ptr = lb.data_ptr<double>();
    for (decltype(numel) i = 0; i != numel; ++i) {
        if (x_ptr[i] < lb_ptr[i]) {
            x_ptr[i] = lb_ptr[i];
        } else if (x_ptr[i] > ub) {
            x_ptr[i] = ub;
        }
    }

    return x/scaling;
}

torch::Tensor clamped_linear(torch::Tensor& x, torch::Tensor lb, torch::Tensor ub, double scaling) {
    lb *= scaling;
    ub *= scaling;

    auto numel = x.numel();
    if (numel != lb.numel()) {
        throw std::logic_error("clamped_linear: x.numel() != lb.numel()");
    }
    if (numel != ub.numel()) {
        throw std::logic_error("clamped_linear: x.numel() != ub.numel()");
    }

    auto x_ptr = x.data_ptr<double>();
    auto lb_ptr = lb.data_ptr<double>();
    auto ub_ptr = ub.data_ptr<double>();
    for (decltype(numel) i = 0; i != numel; ++i) {
        if (x_ptr[i] < lb_ptr[i]) {
            x_ptr[i] = lb_ptr[i];
        } else if (x_ptr[i] > ub_ptr[i]) {
            x_ptr[i] = ub_ptr[i];
        }
    }

    return x/scaling;
}

torch::Tensor clamped_linear_inv(const torch::Tensor& y, double scaling) {
    return y*scaling;
}

torch::Tensor clamped_sigmoid_unbounded_above(torch::Tensor& x, double lb, double soft_ub, double scaling) {
    /*
    auto x_ptr = x.data_ptr<double>();
    for (int64_t i = 0; i != x.numel(); ++i) {
        if (x_ptr[i] < 0.0) { x_ptr[i] = 0.0; }
    }
    */

    return sigmoid_unbounded(x/scaling, 2*lb - soft_ub, soft_ub);
}

torch::Tensor clamped_sigmoid_unbounded_above_inv(const torch::Tensor& y, double lb, double soft_ub, double scaling) {
    return (sigmoid_unbounded_inv(y, 2*lb - soft_ub, soft_ub).clamp_min(0.0))*scaling;
}

void clamped_sigmoid_unbounded_above_enforce_parameter_space(torch::Tensor& x) {
    enforce_parameter_space(x, 0.0);
}

