#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/Quadratic.hpp>

class Quadratic : public Distribution {
    public:
        Quadratic(
            std::string name_in,
            std::shared_ptr<Distribution> dist_in,
            double intercept_in,
            torch::OrderedDict<std::string, torch::Tensor> linear_coefficients_in,
            torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> quadratic_coefficients_in
        ):
            name(std::move(name_in)),
            dist(std::move(dist_in)),
            intercept(intercept_in),
            linear_coefficients(std::move(linear_coefficients_in)),
            quadratic_coefficients(std::move(quadratic_coefficients_in))
        { }

        torch::OrderedDict<std::string, torch::Tensor> generate(
            int64_t sample_size,
            int64_t burn_in_size,
            double first_draw
        ) const override {
            auto out_tensor = torch::full({sample_size}, intercept, torch::kDouble);
            auto out_tensor_a = out_tensor.accessor<double, 1>();
            for (int64_t i = 0; i != sample_size; ++i) {
                auto dist_draw_i = dist->generate(1, 0, 0.0); // We need to change this once we fix NormalVector to use the sample_size.
                for (const auto& item_j : linear_coefficients) {
                    out_tensor_a[i] += torch::dot(dist_draw_i[item_j.key()], item_j.value()).item<double>();
                }
                for (const auto& item_j : quadratic_coefficients) {
                    const auto& key_j = item_j.key();
                    const auto& dist_draw_ij = dist_draw_i[key_j];
                    const auto& quadratic_coefficients_j = item_j.value();
                    for (const auto& item_k : quadratic_coefficients_j) {
                        const auto& key_k = item_k.key();
                        const auto& dist_draw_ik = dist_draw_i[key_k];
                        const auto& quadratic_coefficients_jk = item_k.value();
                        out_tensor_a[i] += torch::dot(dist_draw_ij, torch::matmul(quadratic_coefficients_jk, dist_draw_ik)).item<double>();
                    }
                }
            }
            return {{name, std::move(out_tensor)}};
        }

    private:
        std::string name;
        std::shared_ptr<Distribution> dist;
        double intercept;
        torch::OrderedDict<std::string, torch::Tensor> linear_coefficients;
        torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> quadratic_coefficients;
};

std::unique_ptr<Distribution> ManufactureQuadratic(
    std::string name,
    std::shared_ptr<Distribution> dist,
    double intercept,
    torch::OrderedDict<std::string, torch::Tensor> linear_coefficients,
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> quadratic_coefficients
) {
    return std::make_unique<Quadratic>(
        std::move(name),
        std::move(dist),
        intercept,
        std::move(linear_coefficients),
        std::move(quadratic_coefficients)
    );
}

std::unique_ptr<Distribution> ManufactureQuadratic(
    std::string name,
    std::shared_ptr<Distribution> dist,
    torch::Tensor intercept,
    torch::OrderedDict<std::string, torch::Tensor> linear_coefficients,
    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> quadratic_coefficients
) {
    return ManufactureQuadratic(
        std::move(name),
        std::move(dist),
        intercept.item<double>(),
        std::move(linear_coefficients),
        std::move(quadratic_coefficients)
    );
}

