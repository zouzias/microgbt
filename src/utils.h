#pragma once
#include <utility>
#include<vector>


namespace microgbt {

    using Vector = std::vector<double>;

    class Permutation {
        std::vector<size_t> _perm;
    public:

        explicit Permutation() = default;

        explicit Permutation(std::vector<size_t> permVector): _perm(std::move(permVector)){}

        inline size_t operator()(size_t index) const{
            return _perm[index];
        }
    };
} // namespace microgbt