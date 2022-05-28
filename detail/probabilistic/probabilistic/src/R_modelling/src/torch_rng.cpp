#include <Rinternals.h>
#include <R_support/handle_exception.hpp>
#include <R_support/memory.hpp>
#include <R_protect_guard.hpp>
#include <torch/torch.h>
#include <R_modelling/torch_rng.hpp>

// See Note [Acquire lock when using random generators]
// in pytorch source code (grep for it).

SEXP R_seed_torch_rng(SEXP seed_R) { return R_handle_exception([&](){
    int64_t seed = INTEGER(seed_R)[0];
    torch::manual_seed(seed);
    return R_NilValue;
});}

SEXP R_get_state_torch_rng(void) { return R_handle_exception([&](){
    R_protect_guard protect_guard;
    auto state = []() {
        auto gen = torch::globalContext().defaultGenerator(torch::DeviceType::CPU);
        std::lock_guard<std::mutex> lock(gen.mutex());
        return gen.get_state();
    }();
    auto state_numel = state.numel();
    auto state_a = state.accessor<uint8_t,1>();
    SEXP state_R = protect_guard.protect(Rf_allocVector(RAWSXP, state_numel));
    auto *state_R_a = RAW(state_R);
    for (decltype(state_numel) i = 0; i != state_numel; ++i) {
        state_R_a[i] = state_a[i];
    }
    return state_R;
});}

SEXP R_set_state_torch_rng(SEXP state_R) { return R_handle_exception([&](){
    auto *state_R_a = RAW(state_R);
    auto state_numel = Rf_length(state_R);
    auto state = torch::empty({state_numel}, torch::kByte);
    auto state_a = state.accessor<uint8_t,1>();
    for (decltype(state_numel) i = 0; i != state_numel; ++i) {
        state_a[i] = state_R_a[i];
    }
    auto gen = torch::globalContext().defaultGenerator(torch::DeviceType::CPU);
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_state(state);
    return R_NilValue;
});}

