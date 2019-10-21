#pragma once
#include<vector>


namespace microgbt {

    using Vector = std::vector<double>;

    static double sum(const Vector& vector) {
        size_t n = vector.size();
        double accumulate = 0.0;
        for (size_t i = 0; i < n; i ++){
            accumulate += vector[i];
        }

        return accumulate;
    }

    class Permutation {
        std::vector<size_t> _perm, _inverse;
    public:

        explicit Permutation() = default;

        explicit Permutation(const std::vector<size_t> &permVector):
        _perm(permVector),
        _inverse(_perm.size()){
            for (size_t i = 0 ; i < _perm.size(); i++){
                _inverse[_perm[i]] = i;
            }
        }

        constexpr size_t operator()(size_t index) const{
            return _perm[index];
        }

        constexpr size_t inverse(size_t index) const{
            return _inverse[index];
        }
    };
} // namespace microgbt