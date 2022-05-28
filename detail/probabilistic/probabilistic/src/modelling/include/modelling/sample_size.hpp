#ifndef PROBABILISTIC_SAMPLE_SIZE_HPP_GUARD
#define PROBABILISTIC_SAMPLE_SIZE_HPP_GUARD

#include <cstdint>
#include <string>
#include <torch/torch.h>

int64_t get_sample_size_snn(const torch::OrderedDict<std::string, torch::Tensor>& scores_not_na);
int64_t get_sample_size(const torch::OrderedDict<std::string, torch::Tensor>& scores);

#endif

