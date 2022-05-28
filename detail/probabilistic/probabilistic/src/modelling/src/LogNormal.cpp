#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <log/trivial.hpp>
#include <Rinternals.h>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/LogNormal.hpp>

constexpr double inv_sqrt_2_pi = 0.3989422804014326779399460599343818684758586311649346576659258296;
constexpr double log_2_pi = 1.8378770664093454835606594728112352797227949472755668256343030809;

class LogNormal : public Distribution {
    public:
        LogNormal(
            torch::OrderedDict<std::string, torch::Tensor> mu_in,
            torch::OrderedDict<std::string, torch::Tensor> sigma_in
        ):
            mu(std::move(mu_in)),
            sigma(std::move(sigma_in))
        { }

        LogNormal(const torch::OrderedDict<std::string, torch::Tensor>& log_normals) {
            for (const auto& item : log_normals) {
                const auto& log_normal_name = item.key();
                const auto& log_normal_tensor = item.value();
                mu.insert(log_normal_name, log_normal_tensor.index({torch::indexing::Ellipsis, 0}).squeeze());
                sigma.insert(log_normal_name, log_normal_tensor.index({torch::indexing::Ellipsis, 1}).squeeze());
            }
        }

        torch::OrderedDict<std::string, torch::Tensor> log_density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            torch::OrderedDict<std::string, torch::Tensor> log_likelihoods;
            log_likelihoods.reserve(std::min(mu.size(), observations.size()));
            for (const auto& item : observations) {
                const auto& obs_i_name = item.key();

                const auto *mu_i_ptr = mu.find(obs_i_name);
                if (!mu_i_ptr) { continue; }

                const auto& mu_i = *mu_i_ptr;
                const auto& sigma_i = sigma[obs_i_name];

                const auto& obs_i = item.value();
                auto obs_i_ndim = obs_i.ndimension();

                std::vector<torch::indexing::TensorIndex> common_indices;
                common_indices.reserve(obs_i_ndim);
                for (int64_t j = 0; j != obs_i_ndim; ++j) {
                    common_indices.emplace_back(torch::indexing::Slice(0, std::min(obs_i.sizes().at(j), mu_i.sizes().at(j))));
                }

                auto observations_i_common_indices = obs_i.index(common_indices);
                auto mu_i_common_indices = mu_i.index(common_indices);
                auto sigma_i_common_indices = sigma_i.index(common_indices);

                auto present_indices = torch::logical_or(
                    missing::isna(observations_i_common_indices),
                    missing::isna(mu_i_common_indices)).logical_or(
                    missing::isna(sigma_i_common_indices)
                ).logical_not();
                
                auto observations_i_present_indices = observations_i_common_indices.new_full(
                    observations_i_common_indices.sizes(),
                    missing::na
                );
                observations_i_present_indices.index_put_(
                    {present_indices},
                    observations_i_common_indices.index({present_indices})
                );  

                auto log_observations = [&]() {
                    try {
                        return missing::handle_na(torch::log, observations_i_present_indices);
                    } catch (...) {
                        auto obs_nonpos = torch::logical_and(
                            missing::isna(observations_i_present_indices).logical_not(),
                            observations_i_present_indices.le(0.0)
                        );
                        PROBABILISTIC_LOG_TRIVIAL_DEBUG << observations_i_present_indices.index({obs_nonpos}) << " = nonpositive observations\n"
                                                        << mu_i_common_indices.index({obs_nonpos}) << " = mu.index(obs_nonpos)\n"
                                                        << sigma_i_common_indices.index({obs_nonpos}) << " = sigma.index(obs_nonpos)\n"
                                                        << torch::nonzero({obs_nonpos}) << " = nonpositive observations indices\n";
                        throw;
                    }
                }();

                auto log_observations_minus_means_squared = missing::handle_na(
                    [](const torch::Tensor& lhs, const torch::Tensor& rhs) {
                        return (lhs - rhs).square();
                    },
                    log_observations,
                    mu_i_common_indices
                );

                auto studentised_log_observations_squared = missing::handle_na(
                    [](const torch::Tensor& lhs, const torch::Tensor& rhs) {
                        return lhs/(rhs*rhs);
                    },
                    log_observations_minus_means_squared,
                    sigma_i_common_indices
                );

                auto log_likelihoods_i = missing::handle_na(
                    [](
                        const torch::Tensor& log_obs,
                        const torch::Tensor& sig,
                        const torch::Tensor& std_log_obs_sq
                    ) {
                        return -log_obs - sig.log() - 0.5*(log_2_pi + std_log_obs_sq);
                    },
                    log_observations,
                    sigma_i_common_indices,
                    studentised_log_observations_squared
                );

                #ifndef NDEBUG
                    if (static_cast<torch::Tensor>(log_likelihoods_i.isfinite().all().logical_not()).item<bool>()) {
                        throw std::logic_error("log_likelihoods.isfinite().all().logical_not().item<bool>()");
                    }
                #endif

                log_likelihoods.insert(obs_i_name, std::move(log_likelihoods_i));
            }

            return log_likelihoods;
        }

        torch::OrderedDict<std::string, torch::Tensor> get(void) const override {
            torch::OrderedDict<std::string, torch::Tensor> ret;
            auto mu_size = mu.size();
            ret.reserve(mu_size);
            for (const auto& item : mu) {
                const auto& name = item.key();
                const auto& mu_i = item.value();
                const auto& sigma_i = sigma[name];

                auto ndim_out = mu_i.ndimension() + 1;

                std::vector<int64_t> view_dimensions; view_dimensions.reserve(ndim_out);
                for (int64_t j = 0; j != mu_i.ndimension(); ++j) {
                    view_dimensions.emplace_back(mu_i.sizes().at(j));
                }
                view_dimensions.emplace_back(1);

                ret.insert(name, torch::cat({mu_i.view(view_dimensions), sigma_i.view(view_dimensions)}, ndim_out-1));
            }
            return ret;
        }

        const char * R_dist_function(void) const override {
            static const char fn[] = "function(mu, sigma) { distributional::dist_wrap(\"lnorm\", mu, sigma) }";
            return fn;
        }

    private:
        torch::OrderedDict<std::string, torch::Tensor> mu;
        torch::OrderedDict<std::string, torch::Tensor> sigma;
};

std::unique_ptr<Distribution> ManufactureLogNormal(
    torch::OrderedDict<std::string, torch::Tensor> mu,
    torch::OrderedDict<std::string, torch::Tensor> sigma
) {
    return std::make_unique<LogNormal>(std::move(mu), std::move(sigma));
}

std::unique_ptr<Distribution> ManufactureLogNormal(
    const torch::OrderedDict<std::string, torch::Tensor>& tensors
) {
    return std::make_unique<LogNormal>(tensors);
}

