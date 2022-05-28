#ifndef PROBABILISTIC_STD_SPECIALISATIONS_HASH_HPP_GUARD
#define PROBABILISTIC_STD_SPECIALISATIONS_HASH_HPP_GUARD

// Define hashes for pairs so that pairs may be used as keys in std::unordered_map.
template<class T1, class T2>
struct std::hash<std::pair<T1, T2>> {
    std::size_t operator()(const std::pair<T1,T2>& p) const {
        std::hash<T1> hash_T1;
        std::hash<T2> hash_T2;
        return hash_T1(p.first) ^ hash_T2(p.second);
    }
};

#endif

