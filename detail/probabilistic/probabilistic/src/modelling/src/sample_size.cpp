#include <algorithm>
#include <vector>
#include <cstdint>
#include <torch/torch.h>
#include <libtorch_support/missing.hpp>

int64_t get_sample_size_snn(const torch::OrderedDict<std::string, torch::Tensor>& scores_not_na) {
    int64_t sample_size = 0;
    for (const auto& snn: scores_not_na) {
        sample_size += snn.value().sum(torch::kLong).item<int64_t>();
    }
    return sample_size;
}

int64_t get_sample_size(const torch::OrderedDict<std::string, torch::Tensor>& scores) {
    torch::OrderedDict<std::string, torch::Tensor> scores_not_na;
    scores_not_na.reserve(scores.size());
    for (const auto& s: scores) {
        auto scores_not_na_i = missing::isna(s.value()).logical_not();
        scores_not_na.insert(s.key(), scores_not_na_i);
    }
    return get_sample_size_snn(scores_not_na);
}

