#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <torch/torch.h>
#include <libtorch_support/reparameterisations.hpp>
#include <libtorch_support/Parameterisation.hpp>

#include <log/trivial.hpp>

// If a parameter_in lies outside (lower_bound, upper_bound), place the parameter on the closest boundary
// of [lower_bound + eps*(upper_bound - lower_bound), upper_bound - eps*(upper_bound - lower_bound)].
constexpr double eps = 0.01;

torch::Tensor assert_finite(torch::Tensor x) {
    if (!static_cast<torch::Tensor>(x.isfinite().all()).item<bool>()) {
        throw std::runtime_error("Parameterisation.cpp - assert_finite failed.");
    }

    return x;
}

inline ShapelyParameter disabled_shapely_parameter(void) {
    ShapelyParameter out;
    out.enable = false;
    return out;
}

torch::Tensor Parameterisation::barrier(void) const {
    return torch::full({1}, 0.0, torch::kDouble);
}

torch::Tensor Parameterisation::get_prev(void) const {
    return parameter_on_paper;
}

Parameterisation::~Parameterisation() = default;

Linear::Linear() = default;

Linear::Linear(
    torch::Tensor is_enabled_in,
    std::string name_in,
    torch::Tensor parameter_in,
    torch::Tensor soft_lower_bound_in,
    torch::Tensor soft_upper_bound_in,
    torch::Tensor parameter_scaling_in,
    torch::Tensor barrier_scaling_in
):
    is_enabled(std::move(is_enabled_in)),
    parameter_name(std::move(name_in)),
    parameter(std::move(parameter_in)),
    soft_lower_bound(std::move(soft_lower_bound_in)),
    soft_upper_bound(std::move(soft_upper_bound_in)),
    parameter_scaling(std::move(parameter_scaling_in)),
    barrier_scaling(std::move(barrier_scaling_in))
{
    auto numel = parameter.numel();
    if (is_enabled.item<bool>() && numel) {
        auto ps = parameter_scaling.item<double>();

        auto *param_ptr = parameter.data_ptr<double>();
        for (decltype(numel) i = 0; i != numel; ++i) {
            param_ptr[i] *= ps;
        }
    }
}

bool Linear::enabled(void) const {
    return is_enabled.item<bool>();
}

std::string Linear::name(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Linear::name called, but parameter disabled.");
    }

    return parameter_name;
}

torch::Tensor Linear::get(void) {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Linear::get called, but parameter disabled.");
    }
    
    parameter_on_paper = parameter/parameter_scaling.item<double>();
    return parameter_on_paper;
}

torch::Tensor Linear::barrier(void) const {
    if (is_enabled.item<bool>()) {
        auto bs = barrier_scaling.item<double>();
        if (bs == 0.0) {
            return Parameterisation::barrier();
        }

        auto slb = soft_lower_bound.item<double>();
        auto sub = soft_upper_bound.item<double>();

        auto submslb = sub - slb;
        auto submslb2 = submslb*submslb;
        auto submslb4 = submslb2*submslb2;

        return assert_finite(-bs*16.0*(parameter_on_paper - 0.5*(slb+sub)).square().square()/submslb4);
    } else {
        return Parameterisation::barrier();
    }
}

double Linear::get_soft_lower_bound(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Linear::get_soft_lower_bound called, but parameter disabled.");
    }

    return soft_lower_bound.item<double>();
}

double Linear::get_soft_upper_bound(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Linear::get_soft_upper_bound called, but parameter disabled.");
    }

    return soft_upper_bound.item<double>();
}

double Linear::get_parameter_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Linear::get_parameter_scaling called, but parameter disabled.");
    }

    return parameter_scaling.item<double>();
}

double Linear::get_barrier_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Linear::get_parameter_scaling called, but parameter disabled.");
    }

    return barrier_scaling.item<double>();
}

ShapelyParameter Linear::shapely_parameter_clone(void) {
    torch::NoGradGuard no_grad;
    auto en = enabled();
    if (en) {
        return {
            get(),
            get_soft_lower_bound(),
            get_soft_upper_bound(),
            get_parameter_scaling(),
            get_barrier_scaling(),
            en
        };
    } else {
        return disabled_shapely_parameter();
    }
}

Sigmoid::Sigmoid() = default;

Sigmoid::Sigmoid(
    torch::Tensor is_enabled_in,
    std::string name_in,
    torch::Tensor parameter_in,
    torch::Tensor lower_bound_in,
    torch::Tensor upper_bound_in,
    torch::Tensor parameter_scaling_in,
    torch::Tensor barrier_scaling_in
):
    is_enabled(std::move(is_enabled_in)),
    parameter_name(std::move(name_in)),
    parameter(std::move(parameter_in)),
    lower_bound(std::move(lower_bound_in)),
    upper_bound(std::move(upper_bound_in)),
    parameter_scaling(std::move(parameter_scaling_in)),
    barrier_scaling(std::move(barrier_scaling_in))
{
    auto numel = parameter.numel();
    if (is_enabled.item<bool>() && numel) {
        auto lb = lower_bound.item<double>();
        auto ub = upper_bound.item<double>();
        auto ps = parameter_scaling.item<double>();
        auto delta = ub - lb;

        auto *param_ptr = parameter.data_ptr<double>();
        for (decltype(numel) i = 0; i != numel; ++i) {
            auto param_ptr_i = param_ptr[i];

            if (param_ptr_i <= lb) {
                param_ptr_i = lb + eps*delta;
            } else if (param_ptr_i >= ub) {
                param_ptr_i = ub - eps*delta;
            }

            auto y = (param_ptr_i - lb)/delta;
            param_ptr_i = -log((1.0-y)/y)*ps;

            param_ptr[i] = param_ptr_i;
        }
    }
}

bool Sigmoid::enabled(void) const {
    return is_enabled.item<bool>();
}

std::string Sigmoid::name(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Sigmoid::name called, but parameter disabled.");
    }

    return parameter_name;
}

torch::Tensor Sigmoid::get(void) {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Sigmoid::get called, but parameter disabled.");
    }

    auto lb = lower_bound.item<double>();
    auto ub = upper_bound.item<double>();
    auto ps = parameter_scaling.item<double>();

    parameter_on_paper = lb + (ub - lb)*(parameter/ps).sigmoid();
    return parameter_on_paper;
}

torch::Tensor Sigmoid::barrier(void) const {
    if (is_enabled.item<bool>()) {
        auto lb = lower_bound.item<double>();
        auto ub = upper_bound.item<double>();
        auto bs = barrier_scaling.item<double>();

        return assert_finite(bs*((parameter_on_paper - lb).log() + (ub - parameter_on_paper).log()));
    } else {
        return Parameterisation::barrier();
    }
}

double Sigmoid::get_lower_bound(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Sigmoid::get_lower_bound called, but parameter disabled.");
    }

    return lower_bound.item<double>();
}

double Sigmoid::get_upper_bound(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Sigmoid::get_upper_bound called, but parameter disabled.");
    }

    return upper_bound.item<double>();
}

double Sigmoid::get_parameter_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Sigmoid::get_parameter_scaling called, but parameter disabled.");
    }

    return parameter_scaling.item<double>();
}

double Sigmoid::get_barrier_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Sigmoid::get_barrier_scaling called, but parameter disabled.");
    }

    return barrier_scaling.item<double>();
}

ShapelyParameter Sigmoid::shapely_parameter_clone(void) {
    torch::NoGradGuard no_grad;
    auto en = enabled();
    if (en) {
        return {
            get(),
            get_lower_bound(),
            get_upper_bound(),
            get_parameter_scaling(),
            get_barrier_scaling(),
            en
        };
    } else {
        return disabled_shapely_parameter();
    }
}

SigmoidUnboundedAbove::SigmoidUnboundedAbove() = default;

SigmoidUnboundedAbove::SigmoidUnboundedAbove(
    torch::Tensor is_enabled_in,
    std::string name_in,
    torch::Tensor parameter_in,
    torch::Tensor lower_bound_in,
    torch::Tensor soft_upper_bound_in,
    torch::Tensor parameter_scaling_in,
    torch::Tensor barrier_scaling_in
):
    is_enabled(std::move(is_enabled_in)),
    parameter_name(std::move(name_in)),
    parameter(std::move(parameter_in)),
    lower_bound(std::move(lower_bound_in)),
    soft_upper_bound(std::move(soft_upper_bound_in)),
    parameter_scaling(std::move(parameter_scaling_in)),
    barrier_scaling(std::move(barrier_scaling_in))
{
    auto numel = parameter.numel();
    if (is_enabled.item<bool>() && numel) {
        auto lb = lower_bound.item<double>();
        auto ub = soft_upper_bound.item<double>();
        auto ps = parameter_scaling.item<double>();
        auto delta = ub - lb;

        auto *param_ptr = parameter.data_ptr<double>();
        for (decltype(numel) i = 0; i != numel; ++i) {
            auto param_ptr_i = param_ptr[i];

            if (param_ptr_i <= lb) {
                param_ptr_i = lb + eps*delta;
            }

            param_ptr[i] = delta*ps*sigmoid_unbounded_above_inv(3.0*(param_ptr_i - lb)/(2.0*delta))/3.0;
        }
    }
}

bool SigmoidUnboundedAbove::enabled(void) const {
    return is_enabled.item<bool>();
}

std::string SigmoidUnboundedAbove::name(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnboundedAbove::name called, but parameter disabled.");
    }

    return parameter_name;
}

torch::Tensor SigmoidUnboundedAbove::get(void) {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnboundedAbove::get called, but parameter disabled.");
    }

    auto lb = lower_bound.item<double>();

    parameter_on_paper = sigmoid_unbounded_above(
        parameter,
        lb,
        soft_upper_bound.item<double>(),
        parameter_scaling.item<double>()
    );

    return parameter_on_paper;
}

torch::Tensor SigmoidUnboundedAbove::barrier(void) const {
    if (is_enabled.item<bool>()) {
        auto bs = barrier_scaling.item<double>();
        if (bs == 0.0) {
            return Parameterisation::barrier();
        } 

        auto lb = lower_bound.item<double>();
        auto sub = soft_upper_bound.item<double>();

        auto submlb = sub - lb;
        auto submlb2 = submlb*submlb;
        auto submlb4 = submlb2*submlb2;
        return assert_finite(bs*((parameter_on_paper - lb).log() - 16.0*(parameter_on_paper - 0.5*(lb+sub)).square().square()/submlb4));
    } else {
        return Parameterisation::barrier();
    }
}

double SigmoidUnboundedAbove::get_lower_bound(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnboundedAbove::get_lower_bound called, but parameter disabled.");
    }

    return lower_bound.item<double>();
}

double SigmoidUnboundedAbove::get_soft_upper_bound(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnboundedAbove::get_soft_upper_bound called, but parameter disabled.");
    }

    return soft_upper_bound.item<double>();
}

double SigmoidUnboundedAbove::get_parameter_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnboundedAbove::get_parameter_scaling called, but parameter disabled.");
    }

    return parameter_scaling.item<double>();
}

double SigmoidUnboundedAbove::get_barrier_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnboundedAbove::get_parameter_scaling called, but parameter disabled.");
    }

    return barrier_scaling.item<double>();
}

ShapelyParameter SigmoidUnboundedAbove::shapely_parameter_clone(void) {
    torch::NoGradGuard no_grad;
    auto en = enabled();
    if (en) {
        return {
            get(),
            get_lower_bound(),
            get_soft_upper_bound(),
            get_parameter_scaling(),
            get_barrier_scaling(),
            en
        };
    } else {
        return disabled_shapely_parameter();
    }
}

SigmoidUnbounded::SigmoidUnbounded() = default;

SigmoidUnbounded::SigmoidUnbounded(
    torch::Tensor is_enabled_in,
    std::string name_in,
    torch::Tensor parameter_in,
    torch::Tensor soft_lower_bound_in,
    torch::Tensor soft_upper_bound_in,
    torch::Tensor parameter_scaling_in,
    torch::Tensor barrier_scaling_in
):
    is_enabled(std::move(is_enabled_in)),
    parameter_name(std::move(name_in)),
    parameter(std::move(parameter_in)),
    soft_lower_bound(std::move(soft_lower_bound_in)),
    soft_upper_bound(std::move(soft_upper_bound_in)),
    parameter_scaling(std::move(parameter_scaling_in)),
    barrier_scaling(std::move(barrier_scaling_in))
{
    auto numel = parameter.numel();
    if (is_enabled.item<bool>() && numel) {
        auto lb = soft_lower_bound.item<double>();
        auto ub = soft_upper_bound.item<double>();
        auto ps = parameter_scaling.item<double>();

        auto *param_ptr = parameter.data_ptr<double>();
        for (decltype(numel) i = 0; i != numel; ++i) {
            param_ptr[i] = sigmoid_unbounded_inv(param_ptr[i], lb, ub, ps);
        }
    }
}

bool SigmoidUnbounded::enabled(void) const {
    return is_enabled.item<bool>();
}

std::string SigmoidUnbounded::name(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnbounded::name called, but parameter disabled.");
    }

    return parameter_name;
}

torch::Tensor SigmoidUnbounded::get(void) {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnbounded::get called, but parameter disabled.");
    }

    parameter_on_paper = sigmoid_unbounded(
        parameter,
        soft_lower_bound.item<double>(),
        soft_upper_bound.item<double>(),
        parameter_scaling.item<double>()
    );

    return parameter_on_paper;
}

torch::Tensor SigmoidUnbounded::barrier(void) const {
    if (is_enabled.item<bool>()) {
        auto bs = barrier_scaling.item<double>();
        if (bs == 0.0) {
            return Parameterisation::barrier();
        }

        auto slb = soft_lower_bound.item<double>();
        auto sub = soft_upper_bound.item<double>();

        auto submslb = sub - slb;
        auto submslb2 = submslb*submslb;
        auto submslb4 = submslb2*submslb2;

        return assert_finite(-bs*16.0*(parameter_on_paper - 0.5*(slb+sub)).square().square()/submslb4);
    } else {
        return Parameterisation::barrier();
    }
}

double SigmoidUnbounded::get_soft_lower_bound(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnbounded::get_soft_lower_bound called, but parameter disabled.");
    }

    return soft_lower_bound.item<double>();
}

double SigmoidUnbounded::get_soft_upper_bound(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnbounded::get_soft_upper_bound called, but parameter disabled.");
    }

    return soft_upper_bound.item<double>();
}

double SigmoidUnbounded::get_parameter_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnbounded::get_parameter_scaling called, but parameter disabled.");
    }

    return parameter_scaling.item<double>();
}

double SigmoidUnbounded::get_barrier_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("SigmoidUnbounded::get_parameter_scaling called, but parameter disabled.");
    }

    return barrier_scaling.item<double>();
}

ShapelyParameter SigmoidUnbounded::shapely_parameter_clone(void) {
    torch::NoGradGuard no_grad;
    auto en = enabled();
    if (en) {
        return {
            get(),
            get_soft_lower_bound(),
            get_soft_upper_bound(),
            get_parameter_scaling(),
            get_barrier_scaling(),
            en
        };
    } else {
        return disabled_shapely_parameter();
    }
}

Simplex::Simplex() = default;

Simplex::Simplex(
    torch::Tensor is_enabled_in,
    std::string name_in,
    torch::Tensor parameter_in,
    torch::Tensor parameter_scaling_in,
    torch::Tensor barrier_scaling_in
):
    is_enabled(std::move(is_enabled_in)),
    parameter_name(std::move(name_in)),
    parameter([&parameter_in, &parameter_scaling_in]() -> torch::Tensor&& {
        if (static_cast<torch::Tensor>(parameter_scaling_in.le(0.0).any()).item<double>()) {
            throw std::logic_error("The input parameter to the Simplex must be a vector of positive weights.");
        }

        bool requires_grad = parameter_in.requires_grad();
        parameter_in.detach_();


        auto parameter_in_normalised = parameter_in/parameter_in.sum();
        auto parameter_in_normalised_a = parameter_in_normalised.accessor<double, 1>();

        auto parameter_out_numel = parameter_in_normalised.numel()-1;
        auto parameter_out = parameter_in_normalised.new_full({parameter_out_numel}, std::numeric_limits<double>::quiet_NaN());
        auto parameter_out_a = parameter_out.accessor<double, 1>();

        auto psinv = 1.0/parameter_scaling_in.item<double>();

        auto log_parameter_in_normalised_back = std::log(parameter_in_normalised_a[parameter_out_numel]);

        for (decltype(parameter_out_numel) i = 0; i != parameter_out_numel; ++i) {
            parameter_out_a[i] = psinv*(std::log(parameter_in_normalised_a[i]) - log_parameter_in_normalised_back);
        }

        parameter_in.set_(parameter_out);

        parameter_in.requires_grad_(requires_grad);

        return std::move(parameter_in);
    }()),
    parameter_scaling(std::move(parameter_scaling_in)),
    barrier_scaling(std::move(barrier_scaling_in))
{ }

Simplex::Simplex(
    torch::Tensor is_enabled_in,
    std::string name_in,
    torch::Tensor parameter_in,
    torch::Tensor soft_lower_bound_in,
    torch::Tensor soft_upper_bound_in,
    torch::Tensor parameter_scaling_in,
    torch::Tensor barrier_scaling_in
):
    Simplex(
        std::move(is_enabled_in),
        std::move(name_in),
        std::move(parameter_in),
        std::move(parameter_scaling_in),
        std::move(barrier_scaling_in)
    )
{ }

bool Simplex::enabled(void) const {
    return is_enabled.item<bool>();
}

std::string Simplex::name(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Simplex::name called, but parameter disabled.");
    }

    return parameter_name;
}

torch::Tensor Simplex::get(void) {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Simplex::get called, but parameter disabled.");
    }

    auto parameter_numel = parameter.numel();

    parameter_on_paper = parameter.new_empty({parameter_numel+1});
    parameter_on_paper.index_put_({torch::indexing::Slice(0, parameter_numel)}, parameter_scaling.item<double>()*parameter);
    parameter_on_paper.index_put_({parameter_numel}, 0.0);
    parameter_on_paper = parameter_on_paper.softmax(0);

    return parameter_on_paper;
}

torch::Tensor Simplex::barrier(void) const {
    if (is_enabled.item<bool>()) {
        return assert_finite(barrier_scaling.item<double>() * parameter_on_paper.log());
    } else {
        return Parameterisation::barrier();
    }
}

double Simplex::get_parameter_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Simplex::get_parameter_scaling called, but parameter disabled.");
    }

    return parameter_scaling.item<double>();
}

double Simplex::get_barrier_scaling(void) const {
    if (!is_enabled.item<bool>()) {
        throw std::logic_error("Simplex::get_barrier_scaling called, but parameter disabled.");
    }

    return barrier_scaling.item<double>();
}

ShapelyParameter Simplex::shapely_parameter_clone(void) {
    torch::NoGradGuard no_grad;
    auto en = enabled();
    if (en) {
        return {
            get(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            get_parameter_scaling(),
            get_barrier_scaling(),
            en
        };
    } else {
        return disabled_shapely_parameter();
    }
}
    
