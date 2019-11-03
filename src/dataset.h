#pragma once
#include <vector>
#include<iostream>
#include <Eigen/Dense>
#include <numeric>
#include <memory>
#include "utils.h"

namespace microgbt {

    using Vector = std::vector<double>;
    using VectorT = std::vector<size_t>;
    using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using SortedMatrixType = std::vector<Permutation>;

    /**
    * Represents a machine learning "design matrix" and target vector, (X, y)
    * where the rows and columns of matrix X represent the samples and features, respectively. y is the target vector
    * to be predicted
    */
    class Dataset {

        /**
         * Design matrix, each row correspond to a sample; each column corresponds to a feature
         */
        std::shared_ptr<MatrixType> _X;

        /**
         * Target vector
         */
        std::shared_ptr<Vector> _y;

        SortedMatrixType _sortedMatrixIdx;

        /**
         * Return sorted indices from an Eigen vector
         * @param v
         * @return
         */
         VectorT sortIndices(long colIndex) const{

            // initialize original index locations
            Eigen::VectorXd v = col(colIndex);
            unsigned int n = v.size();

            VectorT idx(n);
            // idx contains now 0,1,...,v.size() - 1
            std::iota(idx.begin(), idx.end(), 0);

            // sort indexes based on comparing values in v
            std::sort(idx.begin(), idx.end(),
                 [&v](int i1, int i2) {return v[i1] < v[i2];});

            return idx;
        }

        inline Eigen::RowVectorXd col(long colIndex) const {
            return _X->col(colIndex);
        }

    public:

        Dataset() = default;

        Dataset(const MatrixType& X, const Vector &y):
        _sortedMatrixIdx(X.cols()) {
            _X = std::make_shared<MatrixType>(X);
            _y = std::make_shared<Vector>(y);
            for ( long j = 0; j < X.cols(); j++) {
                _sortedMatrixIdx[j] = Permutation(sortIndices(j));
            }
        }


        Dataset(Dataset const &dataset) = default;

        inline long nRows() const {
            return _X->rows();
        }

        inline long numFeatures() const {
            return this->_X->cols();
        }

        inline std::shared_ptr<MatrixType> X() const {
            return _X;
        }

        inline Vector y() const {
            return *_y;
        }

        inline Eigen::RowVectorXd row(long rowIndex) const {
            return _X->row(rowIndex);
        }

        inline double operator()(long rowIndex, long colIndex) const {
            return _X->coeffRef(rowIndex, colIndex);
        }

        /**
         * Returns a sorted vector of indices corresponding to a column
         * @param colIndex Index of column
         * @return
         */
        inline const Permutation& sortedColumnIndices(long colIndex) const {
            return _sortedMatrixIdx[colIndex];
        }
    };
}
