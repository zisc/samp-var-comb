#ifndef PROBABILISTIC_LIBTORCH_SUPPORT_TIME_SERIES_HPP_GUARD
#define PROBABILISTIC_LIBTORCH_SUPPORT_TIME_SERIES_HPP_GUARD

#include <vector>
#include <libtorch_support/missing.hpp>
#include <torch/torch.h>

torch::Tensor lag(const torch::Tensor& x, int64_t t_dim = -1, int64_t order = 1, double na_l = missing::na);
// by default make t_dim the last dimension.

template<class T1, class T2>
torch::OrderedDict<T1, T2> lag(
    const torch::OrderedDict<T1, T2>& x,
    int64_t t_dim = -1,
    int64_t order = 1,
    double na_l = missing::na
) {
    torch::OrderedDict<T1, T2> out; out.reserve(x.size());
    for (const auto& item : x) {
        out.insert(item.key(), lag(item.value(), t_dim, order, na_l));
    }
    return out;
}

template<class T>
std::vector<T> lag(
    const std::vector<T>& x,
    int64_t t_dim = -1,
    int64_t order = 1,
    double na_l = missing::na
) {
    std::vector<T> out; out.reserve(x.size());
    for (const auto& xi : x) {
        out.emplace_back(lag(xi, t_dim, order, na_l));
    }
    return out;
}

torch::Tensor diff(const torch::Tensor& x, int64_t t_dim = -1);
// by default make t_dim the last dimension.

class SampleSplitter {
    public:
        SampleSplitter(int64_t in_sample_times_in, int64_t time_dimension_in = -1):
            time_dimension(time_dimension_in),
            in_sample_times(in_sample_times_in)
        { }

        template<class T>
        torch::OrderedDict<T, torch::Tensor> in_sample(const torch::OrderedDict<T, torch::Tensor>& x) const {
            return extract_by_time_index(x, torch::indexing::Slice(0, in_sample_times));
        }

        template<class T>
        torch::OrderedDict<T, torch::Tensor> out_of_sample(const torch::OrderedDict<T, torch::Tensor>& x) const {
            return extract_by_time_index(x, torch::indexing::Slice(in_sample_times, torch::indexing::None));
        }

        template<class T>
        torch::OrderedDict<T, torch::Tensor> h_steps_ahead(const torch::OrderedDict<T, torch::Tensor>& x, int64_t h = 1) const {
            return extract_by_time_index(x, in_sample_times + h);
        }

    private:
        int64_t in_sample_times;
        int64_t time_dimension;

        template<class T>
        torch::OrderedDict<T, torch::Tensor> extract_by_time_index(const torch::OrderedDict<T, torch::Tensor>& x, torch::indexing::TensorIndex idx) const {
            torch::OrderedDict<T, torch::Tensor> ret;
            for (const auto& item : x) {
                const auto& xi = item.value();
                auto xi_sizes = xi.sizes();
                auto xi_sizes_size = xi_sizes.size();
                auto t_dim = time_dimension >= 0 ? time_dimension : xi_sizes_size + time_dimension;
                if (idx.is_integer() && idx.integer() >= xi_sizes.at(t_dim)) {
                    auto insert_size = xi_sizes.vec();
                    insert_size.at(t_dim) = 1;
                    ret.insert(item.key(), xi.new_full(insert_size, missing::na));
                } else {
                    std::vector<torch::indexing::TensorIndex> indices(xi_sizes_size, torch::indexing::Slice());
                    indices.at(t_dim) = idx;
                    ret.insert(item.key(), xi.index(indices));
                }
            }
            return ret;
        }
};

#endif

