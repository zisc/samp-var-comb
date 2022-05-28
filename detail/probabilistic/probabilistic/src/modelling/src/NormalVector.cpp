#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <std_specialisations/hash.hpp>
#include <torch/torch.h>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/NormalVector.hpp>

// TODO: handle missing elegantly.
//       finish implementing Distribution api for NormalVector.

class NormalVector : public Distribution {
    public:
        NormalVector(
            torch::Tensor mu_in,
            torch::Tensor A_in,
            torch::OrderedDict<std::string, torch::indexing::TensorIndex> indices_in
        ):
            mu(std::move(mu_in)),
            A(std::move(A_in)),
            indices(std::move(indices_in))
        { }

        torch::OrderedDict<std::string, torch::Tensor> generate(
            int64_t sample_size,
            int64_t burn_in_size,
            double first_draw
        ) const {
            // Since we can draw from the multivariate normal (almost) directly,
            // we can disregard burn_in_size and first_draw.
            // We need to change this to respect the sample_size argument.
            auto Z = torch::normal(0.0, 1.0, mu.sizes(), c10::nullopt, torch::kDouble);
            auto X = torch::matmul(A, Z) + mu;
            torch::OrderedDict<std::string, torch::Tensor> out;
            for (const auto& item : indices) {
                out.insert(item.key(), X.index(item.value()));
            }

            return out;
        }

    private:
        torch::Tensor mu;
        torch::Tensor A;
        torch::OrderedDict<std::string, torch::indexing::TensorIndex> indices;
};

/*
std::unique_ptr<Distribution> ManufactureNormalVector(
    torch::OrderedDict<std::string, torch::Tensor> mean,
    torch::OrderedDict<std::pair<std::string, std::string>, torch::Tensor> variance
) {
    torch::OrderedDict<std::string, torch::indexing::TensorIndex> indices;
    int64_t begin = 0;
    int64_t end = begin;
    for (const auto& item : mean) {
        auto sizes = item.value().sizes();
        if (sizes.size() != 1) {
            throw std::logic_error("ManufactureNormalVector: mean contains a `!= 1` dimensional tensor.");
        }
        end = begin + sizes.at(0);
        indices.insert(item.key(), torch::indexing::Slice(begin, end));
        begin = end;
    }

    auto mu = torch::full({end}, std::numeric_limits<double>::quiet_NaN(), torch::kDouble);
    for (const auto& item : mean) {
        mu.index_put_({indices[item.key()]}, item.value());
    }

    auto sigma = torch::full({end, end}, std::numeric_limits<double>::quiet_NaN(), torch::kDouble);
    for (const auto& item : variance) {
        auto key = item.key();
        sigma.index_put_({indices[key.first], indices[key.second]}, item.value());
    }

    auto eig = sigma.symeig(true);
    auto sqrt_sigma = torch::matmul(std::get<1>(eig), std::get<0>(eig).sqrt().diag());

    return ManufactureNormalVectorDetail(
        std::move(mu),
        std::move(sqrt_sigma),
        std::move(indices)
    );
}
*/

std::unique_ptr<Distribution> ManufactureNormalVectorDetail(
    torch::Tensor mu,
    torch::Tensor A,
    torch::OrderedDict<std::string, torch::indexing::TensorIndex> indices
) {
    return std::make_unique<NormalVector>(
        std::move(mu),
        std::move(A),
        std::move(indices)
    );
}

