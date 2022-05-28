#include <vector>
#include <torch/torch.h>
#include <libtorch_support/Buffers.hpp>

Buffers Buffers::clone(void) const {
    Buffers out;
    out.buffers.reserve(buffers.size());
    for (const auto& tensor : buffers) {
        out.buffers.emplace_back(tensor.clone());
    }
    return out;
}

