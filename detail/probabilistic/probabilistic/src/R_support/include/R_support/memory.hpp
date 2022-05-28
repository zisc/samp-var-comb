#ifndef PROBABILISTIC_R_SUPPORT_MEMORY_HPP_GUARD
#define PROBABILISTIC_R_SUPPORT_MEMORY_HPP_GUARD

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <Rinternals.h>
#include <R_protect_guard.hpp>

template<class T, class Deleter>
SEXP unique_ptr_to_EXTPTRSXP_register_finalizer(SEXP extptrsxp) {
    R_RegisterCFinalizerEx(
        extptrsxp,
        [](SEXP ptr_to_delete_R) {
            auto *ptr_to_delete = static_cast<T*>(R_ExternalPtrAddr(ptr_to_delete_R));
            if (ptr_to_delete) {
                Deleter deleter;
                deleter(ptr_to_delete);
                R_ClearExternalPtr(ptr_to_delete_R);
            }
        },
        TRUE
    );
    return extptrsxp;
}

template<class T, class Deleter>
SEXP unique_ptr_to_EXTPTRSXP(std::unique_ptr<T, Deleter>&& ptr_in, R_protect_guard& protect_guard) {
    auto *ptr_to_release = ptr_in.release();
    return unique_ptr_to_EXTPTRSXP_register_finalizer<T, Deleter>(
        protect_guard.protect(R_MakeExternalPtr(ptr_to_release, R_NilValue, R_NilValue))
    );
}

template<class T, class Deleter>
SEXP unique_ptr_to_EXTPTRSXP(SEXP list, int i, std::unique_ptr<T, Deleter>&& ptr_in) {
    SEXP ptr_out = R_MakeExternalPtr(ptr_in.release(), R_NilValue, R_NilValue);
    SET_VECTOR_ELT(list, i, ptr_out);
    return unique_ptr_to_EXTPTRSXP_register_finalizer<T, Deleter>(ptr_out);
}

template<class T>
T* EXTPTRSXP_to_ptr(SEXP extptrsxp) {
    if (TYPEOF(extptrsxp) != EXTPTRSXP) {
        throw std::logic_error("EXTPTRSXP_to_ptr: extptrsxp is not an EXTPTRSXP.");
    }

    return static_cast<T*>(R_ExternalPtrAddr(extptrsxp));
}

template<class Derived, class Base = Derived>
SEXP shared_ptr_to_EXTPTRSXP(const std::shared_ptr<Derived>& shared_ptr_in, R_protect_guard& protect_guard) {
    return unique_ptr_to_EXTPTRSXP(
        std::make_unique<std::shared_ptr<Derived>>(shared_ptr_in),
        protect_guard
    );
}

template<class Derived, class Base = Derived>
SEXP shared_ptr_to_EXTPTRSXP(const std::shared_ptr<Derived>&& shared_ptr_in, R_protect_guard& protect_guard) {
    return unique_ptr_to_EXTPTRSXP(
        std::make_unique<std::shared_ptr<Derived>>(std::move(shared_ptr_in)),
        protect_guard
    );
}

template<class Derived, class Base = Derived>
SEXP shared_ptr_to_EXTPTRSXP(SEXP list, int i, const std::shared_ptr<Derived>& shared_ptr_in) {
    return unique_ptr_to_EXTPTRSXP(
        list,
        i,
        std::make_unique<std::shared_ptr<Derived>>(shared_ptr_in)
    );
}

template<class Derived, class Base = Derived>
SEXP shared_ptr_to_EXTPTRSXP(SEXP list, int i, std::shared_ptr<Derived>&& shared_ptr_in) {
    return unique_ptr_to_EXTPTRSXP(
        list,
        i,
        std::make_unique<std::shared_ptr<Derived>>(std::move(shared_ptr_in))
    );
}

template<class Derived, class Base = Derived, class Deleter>
SEXP shared_ptr_to_EXTPTRSXP(std::unique_ptr<Derived, Deleter>&& unique_ptr_in, R_protect_guard& protect_guard) {
    std::shared_ptr<Derived> shared_ptr_in(std::move(unique_ptr_in));
    return shared_ptr_to_EXTPTRSXP<Derived, Base>(std::move(shared_ptr_in), protect_guard);
}

template<class Derived, class Base = Derived, class Deleter>
SEXP shared_ptr_to_EXTPTRSXP(SEXP list, int i, std::unique_ptr<Derived, Deleter>&& unique_ptr_in) {
    std::shared_ptr<Derived> shared_ptr_in(std::move(unique_ptr_in));
    return shared_ptr_to_EXTPTRSXP<Derived, Base>(list, i, std::move(shared_ptr_in));
}

template<class Derived, class Base = Derived>
std::shared_ptr<Derived> EXTPTRSXP_to_shared_ptr(SEXP extptrsxp) {
    if (TYPEOF(extptrsxp) != EXTPTRSXP) {
        throw std::logic_error("EXTPTRSXP_to_shared_ptr: extptrsxp is not an EXTPTRSXP.");
    }

    return *EXTPTRSXP_to_ptr<std::shared_ptr<Derived>>(extptrsxp);
}

#endif

