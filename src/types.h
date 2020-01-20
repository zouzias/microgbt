#pragma once
#include <vector>
#include <Eigen/Dense>

namespace microgbt {

    using Vector = std::vector<double>;
    using VectorD = std::vector<double>;
    using VectorT = std::vector<size_t>;
    using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

}