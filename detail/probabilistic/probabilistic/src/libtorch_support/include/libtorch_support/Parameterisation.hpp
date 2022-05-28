#ifndef PROBABILISTIC_PARAMETERISATION_HPP_GUARD
#define PROBABILISTIC_PARAMETERISATION_HPP_GUARD

#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <torch/torch.h>

struct ShapelyParameter {
    torch::Tensor parameter;
    double lower_bound = std::numeric_limits<double>::quiet_NaN();
    double upper_bound = std::numeric_limits<double>::quiet_NaN();
    double parameter_scaling = std::numeric_limits<double>::quiet_NaN();
    double barrier_scaling = std::numeric_limits<double>::quiet_NaN();
    bool enable = true;
};

class Parameterisation {
    public:
        virtual bool enabled(void) const = 0;
        virtual std::string name(void) const = 0;
        virtual torch::Tensor get(void) = 0;
        virtual torch::Tensor barrier(void) const;
        torch::Tensor get_prev(void) const;
        virtual ShapelyParameter shapely_parameter_clone(void) = 0;
        virtual ~Parameterisation();

    protected:
        torch::Tensor parameter_on_paper;
};

struct NamedShapelyParameters {
    torch::OrderedDict<std::string, ShapelyParameter> parameters;
    int64_t idx = 0;
};

class Linear : public Parameterisation {
    public:
        Linear();

        Linear(
            torch::Tensor is_enabled_in,
            std::string name_in,
            torch::Tensor parameter_in,
            torch::Tensor lower_bound_in,
            torch::Tensor upper_bound_in,
            torch::Tensor parameter_scaling_in,
            torch::Tensor barrier_scaling_in
        );

        bool enabled(void) const override;

        std::string name(void) const override;

        torch::Tensor get(void) override;

        torch::Tensor barrier(void) const override;

        double get_soft_lower_bound(void) const;

        double get_soft_upper_bound(void) const;

        double get_parameter_scaling(void) const;

        double get_barrier_scaling(void) const;

        ShapelyParameter shapely_parameter_clone(void) override;

    private:
        torch::Tensor is_enabled;
        std::string parameter_name;
        torch::Tensor parameter;
        torch::Tensor soft_lower_bound;
        torch::Tensor soft_upper_bound;
        torch::Tensor parameter_scaling;
        torch::Tensor barrier_scaling;
};

class Sigmoid : public Parameterisation {
    public:
        Sigmoid();

        Sigmoid(
            torch::Tensor is_enabled_in,
            std::string name_in,
            torch::Tensor parameter_in,
            torch::Tensor lower_bound_in,
            torch::Tensor upper_bound_in,
            torch::Tensor parameter_scaling_in,
            torch::Tensor barrier_scaling_in
        );

        bool enabled(void) const override;

        std::string name(void) const override;

        torch::Tensor get(void) override;

        torch::Tensor barrier(void) const override;

        double get_lower_bound(void) const;

        double get_upper_bound(void) const;

        double get_parameter_scaling(void) const;
        
        double get_barrier_scaling(void) const;

        ShapelyParameter shapely_parameter_clone(void) override;

    private:
        torch::Tensor is_enabled;
        std::string parameter_name;
        torch::Tensor parameter;
        torch::Tensor lower_bound;
        torch::Tensor upper_bound;
        torch::Tensor parameter_scaling;
        torch::Tensor barrier_scaling;
};

class SigmoidUnboundedAbove : public Parameterisation {
    public:
        SigmoidUnboundedAbove();

        SigmoidUnboundedAbove(
            torch::Tensor is_enabled_in,
            std::string name_in,
            torch::Tensor parameter_in,
            torch::Tensor lower_bound_in,
            torch::Tensor soft_upper_bound_in,
            torch::Tensor parameter_scaling_in,
            torch::Tensor barrier_scaling_in
        );

        bool enabled(void) const override;

        std::string name(void) const override;

        torch::Tensor get(void) override;

        torch::Tensor barrier(void) const override;

        double get_lower_bound(void) const;

        double get_soft_upper_bound(void) const;

        double get_parameter_scaling(void) const;

        double get_barrier_scaling(void) const;

        ShapelyParameter shapely_parameter_clone(void) override;

    private:
        torch::Tensor is_enabled;
        std::string parameter_name;
        torch::Tensor parameter;
        torch::Tensor lower_bound;
        torch::Tensor soft_upper_bound;
        torch::Tensor parameter_scaling;
        torch::Tensor barrier_scaling;
};

class SigmoidUnbounded : public Parameterisation {
    public:
        SigmoidUnbounded();

        SigmoidUnbounded(
            torch::Tensor is_enabled_in,
            std::string name_in,
            torch::Tensor parameter_in,
            torch::Tensor soft_lower_bound_in,
            torch::Tensor soft_upper_bound_in,
            torch::Tensor parameter_scaling_in,
            torch::Tensor barrier_scaling_in
        );

        bool enabled(void) const override;

        std::string name(void) const override;

        torch::Tensor get(void) override;
        torch::Tensor barrier(void) const override;

        double get_soft_lower_bound(void) const;

        double get_soft_upper_bound(void) const;

        double get_parameter_scaling(void) const;

        double get_barrier_scaling(void) const;

        ShapelyParameter shapely_parameter_clone(void) override;

    private:
        torch::Tensor is_enabled;
        std::string parameter_name;
        torch::Tensor parameter;
        torch::Tensor soft_lower_bound;
        torch::Tensor soft_upper_bound;
        torch::Tensor parameter_scaling;
        torch::Tensor barrier_scaling;
};

class Simplex : public Parameterisation {
    public:
        Simplex();

        Simplex(
            torch::Tensor is_enabled_in,
            std::string name_in,
            torch::Tensor parameter_in,
            torch::Tensor parameter_scaling_in,
            torch::Tensor barrier_scaling_in
        );

        Simplex(
            torch::Tensor is_enabled_in,
            std::string name_in,
            torch::Tensor parameter_in,
            torch::Tensor soft_lower_bound_in,
            torch::Tensor soft_upper_bound_in,
            torch::Tensor parameter_scaling_in,
            torch::Tensor barrier_scaling_in
        );

        bool enabled(void) const override;

        std::string name(void) const override;

        torch::Tensor get(void) override;

        torch::Tensor barrier(void) const override;

        double get_parameter_scaling(void) const;

        double get_barrier_scaling(void) const;

        ShapelyParameter shapely_parameter_clone(void) override;

    private:
        torch::Tensor is_enabled;
        std::string parameter_name;
        torch::Tensor parameter;
        torch::Tensor parameter_scaling;
        torch::Tensor barrier_scaling;
};

#endif

