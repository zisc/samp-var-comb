#ifndef PROBABILISTIC_MODELLING_DISTRIBUTION_HPP_GUARD
#define PROBABILISTIC_MODELLING_DISTRIBUTION_HPP_GUARD

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <initializer_list>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>
#include <R_protect_guard.hpp>
#include <R_rng_guard.hpp>
#include <Rinternals.h>

class Distribution {
    public:
        virtual torch::OrderedDict<std::string, torch::Tensor> density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> density(double observations) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_density(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_density(double observations) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> cdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> cdf(double observations) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_cdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_cdf(double observations) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> ccdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> ccdf(double observations) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_ccdf(
            const torch::OrderedDict<std::string, torch::Tensor>& observations
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_ccdf(double observations) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> quantile(
            const torch::OrderedDict<std::string, torch::Tensor>& probabilities
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> quantile(double probability) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> interval_probability(
            const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
            const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> interval_probability(
            double open_lower_bound,
            double closed_upper_bound
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> interval_complement_probability(
            const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
            const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> interval_complement_probability(
            double open_lower_bound,
            double closed_upper_bound
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_interval_probability(
            const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
            const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_interval_probability(
            double open_lower_bound,
            double closed_upper_bound
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_interval_complement_probability(
            const torch::OrderedDict<std::string, torch::Tensor>& open_lower_bound,
            const torch::OrderedDict<std::string, torch::Tensor>& closed_upper_bound
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> log_interval_complement_probability(
            double open_lower_bound,
            double closed_upper_bound
        ) const;

        virtual torch::OrderedDict<std::string, torch::Tensor> draw(void) const {
            throw std::runtime_error("Distribution::draw unimplemented");
        }

        virtual torch::OrderedDict<std::string, torch::Tensor> generate(int64_t sample_size, int64_t burn_in_size, double first_draw) const {
            throw std::runtime_error("Distribution::generate unimplemented.");
        }

        virtual SEXP to_R_list(R_protect_guard& protect_guard) const {
            return to_R_list(
                R_dist_function(),
                get(),
                protect_guard
            );
        }

        /*
        virtual std::vector<SEXP> to_R_lists(R_protect_guard& protect_guard) const {
            return to_R_lists(
                R_dist_function(),
                get(),
                protect_guard
            );
        }
        */

        virtual torch::OrderedDict<std::string, torch::Tensor> get(void) const {
            throw std::runtime_error("Distribution::get unimplemented.");
        }

        virtual torch::OrderedDict<std::string, std::vector<int64_t>> get_structure(void) const {
            throw std::runtime_error("Distribution::get_structure unimplemented.");
        }

        virtual ~Distribution() { }

        virtual const char * R_dist_function(void) const {
            throw std::runtime_error("Distribution::R_dist_function unimplemented.");
        }

    protected:
        torch::OrderedDict<std::string, torch::Tensor> full_as_observations(
            double x,
            const torch::OrderedDict<std::string, std::vector<int64_t>>& structure
        ) const;

        torch::OrderedDict<std::string, torch::Tensor> full_as_observations(double x) const;

    private:
        SEXP to_R_list(
            const char *R_distributional_dist,
            const torch::OrderedDict<std::string, torch::Tensor>& parameters,
            R_protect_guard& protect_guard
        ) const;
};

#endif

