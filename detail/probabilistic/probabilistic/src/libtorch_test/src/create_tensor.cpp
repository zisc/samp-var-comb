#include <R_ext/Print.h>
#include <torch/torch.h>
#include <chrono>
#include "create_tensor.hpp"

auto get_seconds_since_epoch() {
    const auto now = std::chrono::system_clock::now();
    const auto epoch = now.time_since_epoch();
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch);
    return seconds.count();
}

SEXP libtorch_test_create_tensor(void) {
    torch::manual_seed(get_seconds_since_epoch());
    auto Z = torch::normal(0.0, 1.0, {1}, c10::nullopt, torch::kDouble);
    Rprintf("%lf\n", Z.item<double>());
    return R_NilValue;
}

