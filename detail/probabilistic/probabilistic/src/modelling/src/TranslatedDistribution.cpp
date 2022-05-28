#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <log/trivial.hpp>
#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/TranslatedDistribution.hpp>

class TranslatedDistribution : public Distribution {
    public:
        TranslatedDistribution(
            std::shared_ptr<Distribution> distribution_in,
            torch::OrderedDict<std::string, torch::Tensor> translation_in
        ):
            distribution(std::move(distribution_in)),
            translation(std::move(translation_in))
        { }

        torch::OrderedDict<std::string, torch::Tensor> log_density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            torch::OrderedDict<std::string, torch::Tensor> detranslated_observations;
            detranslated_observations.reserve(std::min(translation.size(), observations.size()));
            for (const auto& item : observations) {
                const auto& obs_i_name = item.key();

                const auto *translation_i_ptr = translation.find(obs_i_name);
                if (!translation_i_ptr) { continue; }

                const auto& translation_i = *translation_i_ptr;

                const auto& obs_i = item.value();
                auto obs_i_ndim = obs_i.ndimension();

                std::vector<torch::indexing::TensorIndex> common_indices;
                common_indices.reserve(obs_i_ndim);
                for (int64_t j = 0; j != obs_i_ndim; ++j) {
                    common_indices.emplace_back(torch::indexing::Slice(0, std::min(obs_i.sizes().at(j), translation_i.sizes().at(j))));
                }

                auto observations_i_common_indices = obs_i.index(common_indices);
                auto translation_i_common_indices = translation_i.index(common_indices);

                detranslated_observations.insert(
                    obs_i_name,
                    missing::handle_na(
                        [](const torch::Tensor& x, const torch::Tensor& y) {
                            return x - y;
                        },
                        observations_i_common_indices,
                        translation_i_common_indices
                    )
                );
            }
            return distribution->log_density(detranslated_observations);
        }

        torch::OrderedDict<std::string, torch::Tensor> get(void) const override {
            torch::OrderedDict<std::string, torch::Tensor> ret;
            auto translation_size = translation.size();
            ret.reserve(translation_size);
            const auto& distribution_get = distribution->get();
            for (const auto& item : translation) {
                const auto& name = item.key();

                const auto* distribution_get_i_ptr = distribution_get.find(name);
                if (!distribution_get_i_ptr) { continue; }

                const auto& distribution_get_i = *distribution_get_i_ptr;
                const auto& translation_i = item.value();

                auto ndim_out = translation_i.ndimension() + 1;

                std::vector<int64_t> view_dimensions; view_dimensions.reserve(ndim_out);
                for (int64_t j = 0; j != translation_i.ndimension(); ++j) {
                    view_dimensions.emplace_back(translation_i.sizes().at(j));
                }
                view_dimensions.emplace_back(1);

                ret.insert(name, torch::cat({translation_i.view(view_dimensions), distribution_get_i}));
            }
            return ret;
        }

        const char * R_dist_function(void) const override {
            static std::string fn;
            std::ostringstream ss;
            ss << "function(t, ...) {"
                  "distributional::dist_wrap("
                    "\"tdist\","
                    "do.call(" << distribution->R_dist_function() << ",list(...)),"
                    "t,"
                    "package = \"probabilistic\""
                  ")}";
            fn = ss.str();
            PROBABILISTIC_LOG_TRIVIAL_DEBUG << ss.str() << '\n';
            return fn.c_str();
        }

    private:
        std::shared_ptr<Distribution> distribution;
        torch::OrderedDict<std::string, torch::Tensor> translation;
};

std::unique_ptr<Distribution> ManufactureTranslatedDistribution(
    std::shared_ptr<Distribution> distribution,
    const torch::OrderedDict<std::string, torch::Tensor>& translation
) {
    torch::OrderedDict<std::string, torch::Tensor> translation_squeezed;
    translation_squeezed.reserve(translation.size());
    for (const auto& item : translation) {
        translation_squeezed.insert(item.key(), item.value().squeeze());
    }
    return std::make_unique<TranslatedDistribution>(
        std::move(distribution),
        std::move(translation_squeezed)
    );
}

