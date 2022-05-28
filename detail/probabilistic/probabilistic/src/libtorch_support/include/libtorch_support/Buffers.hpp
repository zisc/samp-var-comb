#ifndef PROBABILISTIC_LIBTORCH_SUPPORT_BUFFER_HPP_GUARD
#define PROBABILISTIC_LIBTORCH_SUPPORT_BUFFER_HPP_GUARD

#include <cstdint>
#include <vector>
#include <torch/torch.h>

struct Buffers {
    std::vector<torch::Tensor> buffers;
    int64_t idx = 0;

    Buffers clone(void) const;
};

#endif

