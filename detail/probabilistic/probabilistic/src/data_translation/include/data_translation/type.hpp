#ifndef PROBABILISTIC_DATA_TRANSLATION_HPP_GUARD
#define PROBABILISTIC_DATA_TRANSLATION_HPP_GUARD

template<class T>
struct tensor_to_R_type {
    static_assert(false, "tensor_to_R_type undefined for this type");
};

template<> struct tensor_to_R_type<bool>    { typedef int    type; };
template<> struct tensor_to_R_type<int64_t> { typedef int    type; };
template<> struct tensor_to_R_type<double>  { typedef double type; };

template<class T>
using tensor_to_R_type_t = typename tensor_to_R_type<T>::type;

#endif

