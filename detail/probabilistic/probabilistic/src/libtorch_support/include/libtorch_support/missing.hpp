#ifndef PROBABILISTIC_MISSING_HPP_GUARD
#define PROBABILISTIC_MISSING_HPP_GUARD

#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <torch/torch.h>

// Using NaNs to represent missing values in libtorch Tensors always
// results in NaN gradients, even when we index the Tensors to avoid
// the NaNs, and hence even when the loss is not NaN. See this issue:
// https://github.com/pytorch/pytorch/issues/15131. The key insight is
// that all values in tensors that participate in the computation graph
// must have elements that, when multiplied by 0.0, return 0.0. This
// means that we must also avoid Inf and -Inf as well. To handle missing
// values, we will therefore represent them as some other, very small
// value. Then, we define wrappers around libtorch operations that take in
// (a) tensor(s) and return a tensor, and that do the following:
//     1. Replace our na values with NaN in a cloned and detached
//        version of the inputs.
//     2. Do the operation, noting where missing values appear in
//        the output.
//     3. Repeat the computation with the original inputs where
//        our na represents missing.
//     4. Where missing values appear in the output from step 2,
//        change those elements in the output from step 3 to our
//        na value.
// This means that every argument needs to be cloned, and every
// operation must be run twice: once to find the missing values
// in the output, and then a second time so that the arguments
// do not contain NaNs and corrupt the gradients. To avoid this
// cost, we would need to know where the missing values ought to
// appear in the output ahead of time. This differs across
// operations however, and we would need to write a different such
// wrapper for every different kind of operations. For example,
// elementwise operations like exp are one kind of operation,
// matrix multiplication is another. We also can't have na be a
// large number, since we need the operations to be finite when
// applied to elements that a missing. Consider taking the
// exponential of std::numeric_limits<double>::max() for example.
//
// Another solution suggested by the above github issue is to
// register backward hook on tensors and overwrite the gradients
// during the backward pass. However, this doesn't seem to work
// with torch::matmul, possibly because it calls other operations
// with tensors we don't have access to, and therefore to which
// we cannot register a backward hook.

namespace missing {
    
    static constexpr double na = 5.25296e-20;

    // These functions didn't work the way I intended - see missing.cpp.
    /*
    torch::Tensor where(const torch::Tensor& condition, const torch::Tensor& value_if_true, const torch::Tensor& value_if_false);
    torch::Tensor where(const torch::Tensor& condition, const torch::Tensor& value_if_true, const torch::Scalar& value_if_false);
    torch::Tensor where(const torch::Tensor& condition, const torch::Scalar& value_if_true, const torch::Tensor& value_if_false);
    torch::Tensor where(const torch::Tensor& condition, const torch::Scalar& value_if_trie, const torch::Scalar& value_if_false);
    */

    inline torch::Tensor isna(const torch::Tensor& x, double na_arg = na) {
        return x.eq(na_arg);
    }

    inline bool isna(double x, double na_arg = na) {
        return x == na_arg;
    }

    template<class T1, class T2>
    auto isna(const torch::OrderedDict<T1, T2>& x, double na_arg = na) {
        torch::OrderedDict<T1, T2> ret;
        for (const auto& item : x) {
            ret.insert(item.key(), isna(item.value(), na_arg));
        }
        return ret;
    }

    inline torch::Tensor is_present(const torch::Tensor& x, double na_arg = na) { return x.ne(na_arg); }

    inline bool is_present(double x, double na_arg = na) { return !isna(x, na_arg); }

    template<class T1, class T2>
    auto is_present(const torch::OrderedDict<T1, T2>& x, double na_arg = na) {
        torch::OrderedDict<T1, T2> ret;
        for (const auto& item : x) {
            ret.insert(item.key(), is_present(item.value(), na_arg));
        }
        return ret;
    }

    inline torch::Tensor replace_na(const torch::Tensor& x, double na_in, double replacement) {
        if (na_in != replacement) {
            auto x_clone = x.clone();
            x_clone.index_put_({x.eq(na_in)}, replacement);
            return x_clone;
        } else {
            return x;
        }
    }

    inline torch::Tensor replace_na(const torch::Tensor& x, double replacement) {
        return replace_na(x, na, replacement);
    }

    #ifndef NDEBUG
        template<class E>
        void throw_if_true_impl(const char *message, size_t arg_num, bool arg) {
            if (arg) {
                std::ostringstream ss;
                ss << message << " Argument " << arg_num << " failed.";
                throw E(ss.str());
            }
        }

        template<class E, class... T>
        void throw_if_true_impl(const char *message, size_t arg_num, bool arg, T&&... args) {
            throw_if_true_impl<E>(message, arg_num, arg);
            throw_if_true_impl<E>(message, arg_num+1, args...);
        }

        template<class E, class... T>
        void throw_if_true(const char *message, T&&... args) {
            throw_if_true_impl<E>(message, 1, args...);
        }
    #endif

    template<class OP, class Tuple, std::size_t... Is>
    torch::Tensor apply_op_impl(OP&& op, Tuple&& t, std::index_sequence<Is...>) {
        return op(std::get<Is>(std::forward<Tuple>(t))...);
    }

    template<class OP, class Tuple>
    torch::Tensor apply_op(OP&& op, Tuple&& t) {
        return apply_op_impl(
            std::forward<OP>(op),
            std::forward<Tuple>(t),
            std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>{}>{}
        );
    }

    inline void na_to_nan_impl(torch::Tensor& arg) {
        if (arg.scalar_type() == torch::kDouble) {
            arg.index_put_({isna(arg)}, std::numeric_limits<double>::quiet_NaN());
        } else if (arg.scalar_type() == torch::kFloat) {
            arg.index_put_({isna(arg)}, std::numeric_limits<float>::quiet_NaN());
        }
    }

    template<class T, class... Ts>
    void na_to_nan_impl(T&& arg, Ts&... args) {
        na_to_nan_impl(arg);
        na_to_nan_impl(args...);
    }

    template<class Tuple, size_t... Is>
    void na_to_nan_impl(Tuple& t, std::index_sequence<Is...>) {
        na_to_nan_impl(std::get<Is>(t)...);
    }

    template<class Tuple>
    void na_to_nan(Tuple& t) {
        na_to_nan_impl(t, std::make_index_sequence<std::tuple_size<Tuple>{}>{});
    }

    template<class OP, class... T>
    torch::Tensor handle_na(OP&& op, T&&... args) {
        #ifndef NDEBUG
            throw_if_true<std::logic_error>(
                "messing::handle_na: args.isfinite().logical_not().any()...",
                static_cast<torch::Tensor>(args.isfinite().logical_not().any()).item<bool>()...
            );
        #endif

        auto args_detached = std::make_tuple(args.detach().clone()...);
        na_to_nan(args_detached);

        auto op_args_detached = apply_op(op, args_detached).detach();
        auto op_args_detached_nan = op_args_detached.isnan();

        auto op_args = op(args...);

        auto out = op_args.new_empty(op_args.sizes());
        out.index_put_({op_args_detached_nan}, na);
        out.index_put_(
            {op_args_detached_nan.logical_not()},
            op_args.index({op_args_detached_nan.logical_not()})
        );

        #ifndef NDEBUG
            if (static_cast<torch::Tensor>(op_args.isfinite().logical_not().any()).item<bool>()) {
                throw std::logic_error("missing::handle_na: op_args.isfinite().logical_not().any()");
            }

            if (static_cast<torch::Tensor>(out.isfinite().logical_not().any()).item<bool>()) {
                throw std::logic_error("missing::handle_na: out.isfinite().logical_not().any()");
            }
        #endif

        return out;
    }

    template<class OP, class D, class... T>
    torch::Tensor handle_na_debug(OP&& op, D&& debug, T&&... args) {
        #ifndef NDEBUG
            try {
                return handle_na(std::forward<OP>(op), args...);
            } catch(...) {
                debug(std::forward<T>(args)...);
            }
        #else
            return handle_na(std::forward<OP>(op), std::forward<T>(args)...);
        #endif
    }

};

#endif

