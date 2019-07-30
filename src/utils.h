#pragma once
#include<vector>


namespace microgbt {

    using Vector = std::vector<double>;

    static double par_simd_accumulate(const Vector& vector) {
        size_t n = vector.size();
        double accumulate = 0.0;
        for (size_t i = 0; i < n; i ++){
            accumulate  += vector[i];
        }

        return accumulate;
    }
} // namespace microgbt
