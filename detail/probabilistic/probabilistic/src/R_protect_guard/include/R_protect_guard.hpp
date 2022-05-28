#ifndef R_PROTECT_GUARD_HPP_GUARD
#define R_PROTECT_GUARD_HPP_GUARD

#include <Rinternals.h>

class R_protect_guard {
    public:
        SEXP protect(SEXP exp) {
            SEXP ret = PROTECT(exp);
            ++reference_count;
            return ret;
        }

        R_protect_guard(): reference_count(0) { }

        R_protect_guard(const R_protect_guard&) = delete;

        R_protect_guard(R_protect_guard&&) = delete;

        ~R_protect_guard() {
            if (reference_count > 0) {
                UNPROTECT(reference_count);
            }
        }

    private:
        int reference_count;
};

#endif

