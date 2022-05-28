#ifndef R_RNG_GUARD_HPP_GUARD
#define R_RNG_GUARD_HPP_GUARD

#include <R_ext/Random.h>

class R_rng_guard {
    public:
        R_rng_guard() { GetRNGstate(); }
        ~R_rng_guard() { PutRNGstate(); }
};

#endif

