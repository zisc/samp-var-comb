#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Random.h>
#include <R_protect_guard.hpp>
#include <R_rng_guard.hpp>
#include <R_support/function.hpp>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <libtorch_support/indexing.hpp>
#include <libtorch_support/standard_normal_log_cdf.hpp>
#include <modelling/distribution/Distribution.hpp>
#include <modelling/distribution/Normal.hpp>

constexpr double log_2_pi = 1.8378770664093454835606594728112352797227949472755668256343030809;
constexpr double inv_sqrt_2 = 0.7071067811865475244008443621048490392848359376884740365883398689;
constexpr double sqrt_2 = 1.4142135623730950488016887242096980785696718753769480731766797379;

class Normal : public Distribution {
    private:
        // Move this function to somewhere general where it can be shared by various distributions.
        std::vector<torch::indexing::TensorIndex> get_common_indices(
            const torch::Tensor& x,
            const torch::Tensor& y
        ) const {
            auto ndim = x.ndimension();
            if (y.ndimension() != ndim) {
                std::ostringstream ss;
                ss << "x.ndimension() == " << ndim << " != " << y.ndimension() << " == y.ndimenion()";
                throw std::logic_error(ss.str());
            }
            auto x_sizes = x.sizes();
            auto y_sizes = y.sizes();
            std::vector<torch::indexing::TensorIndex> out; out.reserve(ndim);
            for (int64_t j = 0; j != ndim; ++j) {
                out.emplace_back(torch::indexing::Slice(0, std::min(x_sizes.at(j), y_sizes.at(j))));
            }
            return out;
        }

        template<class T>
        struct Common {
            T observations;
            torch::Tensor mean;
            torch::Tensor std_dev;
        };

        auto get_common(
            const torch::Tensor& observations,
            const torch::Tensor& mean,
            const torch::Tensor& std_dev
        ) const {
            auto common_indices = get_common_indices(observations, mean);
            Common<torch::Tensor> common;
            common.observations = observations.index(common_indices);
            common.mean = mean.index(common_indices);
            common.std_dev = std_dev.index(common_indices);
            return common;
        }

        auto get_common(
            double obs,
            const torch::Tensor& mean,
            const torch::Tensor& std_dev
        ) const {
            Common<double> common;
            common.observations = obs;
            common.mean = mean;
            common.std_dev = std_dev;
            return common;
        }

        template <class F, class T>
        auto property_at_obs_dict_elem(
            F&& op,
            const T& obs,
            torch::Tensor mean,
            torch::Tensor std_dev
        ) const {
            auto common = get_common(obs, mean, std_dev);
            return op(common.observations, common.mean, common.std_dev);
        }

        template <class F, class T>
        auto property_at_obs_dict(
            F&& op,
            const torch::OrderedDict<std::string, T>& obs
        ) const {
            torch::OrderedDict<std::string, torch::Tensor> out; out.reserve(mean.size());
            for (const auto& item : mean) {
                const auto& name = item.key();
                const auto& mean_i = item.value();
                const auto& std_dev_i = std_dev[name];
                const auto& obs_i = obs[name];
                out.insert(name, property_at_obs_dict_elem(op, obs_i, mean_i, std_dev_i));
            }
            return out;
        }

        template <class F>
        auto property_at_obs_dict(
            F&& op,
            double obs
        ) const {
            torch::OrderedDict<std::string, torch::Tensor> out; out.reserve(mean.size());
            for (const auto& item : mean) {
                const auto& name = item.key();
                const auto& mean_i = item.value();
                const auto& std_dev_i = std_dev[name];
                out.insert(name, property_at_obs_dict_elem(op, obs, mean_i, std_dev_i));
            }
            return out;
        }

    public:
        Normal(
            torch::OrderedDict<std::string, torch::Tensor> mean_in,
            torch::OrderedDict<std::string, torch::Tensor> std_dev_in
        ):
            mean(std::move(mean_in)),
            std_dev(std::move(std_dev_in))
        {
            std::vector<int64_t> new_sizes;
            for (auto& item : mean) {
                const auto& key = item.key();
                auto& m = item.value();
                auto& s = std_dev[key];

                auto mndim = m.ndimension();
                auto sndim = s.ndimension();

                if (mndim != 0 && mndim != 0 && mndim != sndim) {
                    throw std::logic_error("mndim != 0 && mndim != 0 && mndim != sndim");
                }

                if (mndim == 0) {
                    m = m.expand(s.sizes());
                } else if (sndim == 0) {
                    s = s.expand(m.sizes());
                } else {
                    auto m_sizes = m.sizes();
                    auto s_sizes = s.sizes();
                    new_sizes.reserve(mndim);
                    for (decltype(mndim) j = 0; j != mndim; ++j) {
                        new_sizes.emplace_back(std::max(m_sizes.at(j), s_sizes.at(j)));
                    }
                    m = m.expand(new_sizes);
                    s = s.expand(new_sizes);
                    new_sizes.clear();
                }
            }
        }

        Normal(const torch::OrderedDict<std::string, torch::Tensor>& normals) {
            for (const auto& item : normals) {
                const auto& normal_name = item.key();
                const auto& normal_tensor = item.value();
                mean.insert(normal_name, normal_tensor.index({torch::indexing::Ellipsis, 0}).squeeze());
                std_dev.insert(normal_name, normal_tensor.index({torch::indexing::Ellipsis, 1}).squeeze());
            }
        }

        template <class T>
        torch::OrderedDict<std::string, torch::Tensor> density_impl(
            const T& observations
        ) const {
            return elementwise_unary_op(
                log_density(observations),
                [](const torch::Tensor& x) { return x.exp(); }
            );
        }

        torch::OrderedDict<std::string, torch::Tensor> density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            return density_impl(observations);
        }

        /*
        torch::OrderedDict<std::string, torch::Tensor> density(
            double observations
        ) const override {
            return density_impl(observations);
        }
        */

        /*
        torch::OrderedDict<std::string, torch::Tensor> density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            auto ret = log_density(observations);
            for (auto& item : ret) {
                auto& tensor = item.value();
                tensor = missing::handle_na(
                    [](const torch::Tensor& x) { return x.exp(); },
                    tensor
                );
            }
            return ret;
        }
        */

        template <class T>
        torch::OrderedDict<std::string, torch::Tensor> log_density_impl(
            const T& observations
        ) const {
            return property_at_obs_dict(
                [](const auto& obs, const auto& m, const auto& s) {
                    auto obs_minus_mean_squared = missing::handle_na(
                        [](const auto& lhs, const auto& rhs) {
                            return (lhs - rhs).square();
                        },
                        obs,
                        m
                    );

                    auto studentised_obs_squared = missing::handle_na(
                        [](const auto& lhs, const auto& rhs) {
                            return lhs/(rhs*rhs);
                        },
                        obs_minus_mean_squared,
                        s
                    );

                    return missing::handle_na(
                        [](const auto& std, const auto& std_obs_sq) {
                            return -std.log() - 0.5*(log_2_pi + std_obs_sq);
                        },
                        s,
                        studentised_obs_squared
                    );
                },
                observations
            );
        }

        torch::OrderedDict<std::string, torch::Tensor> log_density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            return log_density_impl(observations);
        }

        /*
        torch::OrderedDict<std::string, torch::Tensor> log_density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            torch::OrderedDict<std::string, torch::Tensor> log_likelihoods;
            log_likelihoods.reserve(std::min(mean.size(), observations.size()));
            for (const auto& item : observations) {
                const auto& obs_i_name = item.key();
                const auto *mean_i_ptr = mean.find(obs_i_name);
                if (!mean_i_ptr) { continue; }
                const auto& mean_i = *mean_i_ptr;
                const auto& std_dev_i = std_dev[obs_i_name];
                const auto& obs_i = item.value();
                auto common = get_common(obs_i, mean_i, std_dev_i);

                auto observations_minus_means_squared = missing::handle_na(
                    [](const torch::Tensor& lhs, const torch::Tensor& rhs) {
                        return (lhs - rhs).square();
                    },
                    common.observations,
                    common.mean
                );

                auto studentised_observations_squared = missing::handle_na(
                    [](const torch::Tensor& lhs, const torch::Tensor& rhs) {
                        return lhs/(rhs*rhs);
                    },
                    observations_minus_means_squared,
                    common.std_dev
                );

                log_likelihoods.insert(obs_i_name, missing::handle_na(
                    [](const torch::Tensor& std, const torch::Tensor& std_obs_sq) {
                        return -std.log() - 0.5*(log_2_pi + std_obs_sq);
                    },
                    common.std_dev,
                    studentised_observations_squared
                ));
            }
            return log_likelihoods;
        }
        */

        /*
         * Doesn't work, since handle_na is called with observations as an argument, and handle_na does not support double.
        torch::OrderedDict<std::string, torch::Tensor> log_density(
            double observations
        ) const override {
            return log_density_impl(observations);
        }
        */

        torch::OrderedDict<std::string, torch::Tensor> cdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            torch::OrderedDict<std::string, torch::Tensor> out;
            out.reserve(std::min(mean.size(), observations.size()));
            for (const auto& item : observations) {
                const auto& obs_i_name = item.key();
                const auto *mean_i_ptr = mean.find(obs_i_name);
                if (!mean_i_ptr) { continue; }
                const auto& mean_i = *mean_i_ptr;
                const auto& std_dev_i = std_dev[obs_i_name];
                const auto& obs_i = item.value();
                auto common = get_common(obs_i, mean_i, std_dev_i);

                auto studentised_observations = missing::handle_na(
                    [](const torch::Tensor& obs, const torch::Tensor& m, const torch::Tensor& s) {
                        return (obs - m)/s;
                    },
                    common.observations,
                    common.mean,
                    common.std_dev
                );

                out.insert(obs_i_name, missing::handle_na(
                    [](const torch::Tensor& z) {
                        return 0.5*torch::erfc(-inv_sqrt_2*z);
                    },
                    studentised_observations
                ));
            }
            return out;
        }

        torch::OrderedDict<std::string, torch::Tensor> log_cdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            torch::OrderedDict<std::string, torch::Tensor> out;
            out.reserve(std::min(mean.size(), observations.size()));
            for (const auto& item : observations) {
                const auto& obs_i_name = item.key();
                const auto *mean_i_ptr = mean.find(obs_i_name);
                if (!mean_i_ptr) { continue; }
                const auto& mean_i = *mean_i_ptr;
                const auto& std_dev_i = std_dev[obs_i_name];
                const auto& obs_i = item.value();
                auto common = get_common(obs_i, mean_i, std_dev_i);

                auto studentised_observations = missing::handle_na(
                    [](const torch::Tensor& obs, const torch::Tensor& m, const torch::Tensor& s) {
                        return (obs - m)/s;
                    },
                    common.observations,
                    common.mean,
                    common.std_dev
                );

                try {
                out.insert(obs_i_name, missing::handle_na(
                    [](const torch::Tensor& z) {
                        return standard_normal_log_cdf(z);
                    },
                    studentised_observations
                ));
                } catch(...) {
                    const auto& z = studentised_observations;
                    std::cout << z << " = z" << std::endl;
                    std::cout << standard_normal_log_cdf(z) << " = standard_normal_log_cdf(z)" << std::endl;
                    throw;
                }
            }
            return out;
        }

        torch::OrderedDict<std::string, torch::Tensor> ccdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            torch::OrderedDict<std::string, torch::Tensor> out;
            out.reserve(std::min(mean.size(), observations.size()));
            for (const auto& item : observations) {
                const auto& obs_i_name = item.key();
                const auto *mean_i_ptr = mean.find(obs_i_name);
                if (!mean_i_ptr) { continue; }
                const auto& mean_i = *mean_i_ptr;
                const auto& std_dev_i = std_dev[obs_i_name];
                const auto& obs_i = item.value();
                auto common = get_common(obs_i, mean_i, std_dev_i);

                auto studentised_observations = missing::handle_na(
                    [](const torch::Tensor& obs, const torch::Tensor& m, const torch::Tensor& s) {
                        return (obs - m)/s;
                    },
                    common.observations,
                    common.mean,
                    common.std_dev
                );

                out.insert(obs_i_name, missing::handle_na(
                    [](const torch::Tensor& z) {
                        return 0.5*torch::erfc(inv_sqrt_2*z);
                    },
                    studentised_observations
                ));
            }
            return out;
        }

        torch::OrderedDict<std::string, torch::Tensor> log_ccdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const override {
            torch::OrderedDict<std::string, torch::Tensor> out;
            out.reserve(std::min(mean.size(), observations.size()));
            for (const auto& item : observations) {
                const auto& obs_i_name = item.key();
                const auto *mean_i_ptr = mean.find(obs_i_name);
                if (!mean_i_ptr) { continue; }
                const auto& mean_i = *mean_i_ptr;
                const auto& std_dev_i = std_dev[obs_i_name];
                const auto& obs_i = item.value();
                auto common = get_common(obs_i, mean_i, std_dev_i);

                auto studentised_observations = missing::handle_na(
                    [](const torch::Tensor& obs, const torch::Tensor& m, const torch::Tensor& s) {
                        return (obs - m)/s;
                    },
                    common.observations,
                    common.mean,
                    common.std_dev
                );

                out.insert(obs_i_name, missing::handle_na(
                    [](const torch::Tensor& z) {
                        return standard_normal_log_cdf(-z);
                    },
                    studentised_observations
                ));
            }
            return out;
        }

        torch::OrderedDict<std::string, torch::Tensor> quantile(
            const torch::OrderedDict<std::string, torch::Tensor>& probabilities
        ) const override {
            torch::OrderedDict<std::string, torch::Tensor> out;
            out.reserve(std::min(mean.size(), probabilities.size()));
            for (const auto& item : probabilities) {
                const auto& probs_i_name = item.key();
                const auto *mean_i_ptr = mean.find(probs_i_name);
                if (!mean_i_ptr) { continue; }
                const auto& mean_i = *mean_i_ptr;
                const auto& std_dev_i = std_dev[probs_i_name];
                const auto& probs_i = item.value();
                auto common = get_common(probs_i, mean_i, std_dev_i);

                out.insert(probs_i_name, missing::handle_na(
                    [](const torch::Tensor& probs, const torch::Tensor& m, const torch::Tensor& s) {
                        return m + sqrt_2*s*torch::erfinv(2*probs-1);
                    },
                    common.observations,
                    common.mean,
                    common.std_dev
                ));
            }
            return out;
        }

        torch::OrderedDict<std::string, torch::Tensor> draw(void) const override {
            torch::OrderedDict<std::string, torch::Tensor> out; out.reserve(mean.size());
            for (const auto& item : mean) {
                const auto& name = item.key();
                const torch::Tensor& mean_i = item.value();
                const torch::Tensor& std_dev_i = std_dev[name];
                out.insert(name, at::normal(mean_i, std_dev_i));
            }
            return out;
        }

        torch::OrderedDict<std::string, torch::Tensor> get(void) const override {
            torch::OrderedDict<std::string, torch::Tensor> ret;
            auto mean_size = mean.size();
            ret.reserve(mean_size);
            for (const auto& item : mean) {
                const auto& name = item.key();
                const auto& mean_i = item.value();
                const auto& std_dev_i = std_dev[name];

                auto ndim_out = mean_i.ndimension() + 1;

                std::vector<int64_t> view_dimensions; view_dimensions.reserve(ndim_out);
                for (int64_t j = 0; j != mean_i.ndimension(); ++j) {
                    view_dimensions.emplace_back(mean_i.sizes().at(j));
                }
                view_dimensions.emplace_back(1);

                ret.insert(name, torch::cat({mean_i.view(view_dimensions), std_dev_i.view(view_dimensions)}, ndim_out-1));
            }
            return ret;
        }

        torch::OrderedDict<std::string, std::vector<int64_t>> get_structure(void) const override {
            torch::OrderedDict<std::string, std::vector<int64_t>> out; out.reserve(mean.size());
            for (const auto& item : mean) {
                out.insert(item.key(), item.value().sizes().vec());
            }
            return out;
        }

        const char * R_dist_function(void) const override {
            static const char fn[] = "distributional::dist_normal";
            return fn;
        }

    private:
        torch::OrderedDict<std::string, torch::Tensor> mean;
        torch::OrderedDict<std::string, torch::Tensor> std_dev;
        // std::vector<int64_t> sizes;
};
                    

std::unique_ptr<Distribution> ManufactureNormal(
    torch::OrderedDict<std::string, torch::Tensor> mean,
    torch::OrderedDict<std::string, torch::Tensor> std_dev
) {
    return std::make_unique<Normal>(std::move(mean), std::move(std_dev));
}

std::unique_ptr<Distribution> ManufactureNormal(const torch::OrderedDict<std::string, torch::Tensor>& tensors) {
    return std::make_unique<Normal>(tensors);
}

