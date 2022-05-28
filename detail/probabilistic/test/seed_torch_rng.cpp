#include <chrono>
#include <torch/torch.h>
#include <seed_torch_rng.hpp>

void seed_torch_rng(void) {
    static bool done = false;
    if (!done) {
        const auto now = std::chrono::system_clock::now();
        const auto epoch = now.time_since_epoch();
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch);
        torch::manual_seed(seconds.count());
        done = true;
    }
}

