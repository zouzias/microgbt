#pragma once
#include <vector>
#include<iostream>
#include <Eigen/Dense>
#include <numeric>
#include <memory>
#include "trees/split_info.h"

namespace microgbt {

    using Vector = std::vector<double>;
    using VectorT = std::vector<size_t>;
    using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    /**
    * Dataset represents a machine learning "design matrix" and target vector, (X, y)
    * where the rows and columns of matrix X represent the samples and features, respectively. y is the target vector
    * to be predicted
    */
    class Dataset {

        // Design matrix, each row corresponds to a sample; each column corresponds to a feature
        std::shared_ptr<MatrixType> _X;

        // Target vector
        std::shared_ptr<Vector> _y;

        std::vector<VectorT> _sortedMatrixIdx;

        VectorT _rowIndices;

        /**
         * Return sorted indices from an Eigen vector
         * @param v
         * @return
         */
         VectorT sortIndices(long colIndex) const{

            // initialize original index locations
            VectorT idx(_rowIndices.size());
            // idx contains now 0,1,...,v.size() - 1
            std::iota(idx.begin(), idx.end(), 0);

            // sort indexes based on comparing values in v
            std::sort(idx.begin(), idx.end(),
                 [this, colIndex](long i1, long i2) {
                return _X->coeffRef(_rowIndices[i1], colIndex) < _X->coeffRef(_rowIndices[i2], colIndex);
            });

            return idx;
        }
    public:

        Dataset() = default;

        Dataset(const MatrixType& X, const Vector &y):
        _sortedMatrixIdx(X.cols()),
        _rowIndices(y.size()){
            _X = std::make_shared<MatrixType>(X);
            _y = std::make_shared<Vector>(y);

            // By default, all rows are included in the dataset
            std::iota(_rowIndices.begin(), _rowIndices.end(), 0);

            for ( long j = 0; j < X.cols(); j++) {
                _sortedMatrixIdx[j] = sortIndices(j);
            }
        }


        Dataset(Dataset const &dataset) = default;

        /**
         * Construct a Dataset, given a binary split gain and lef/right side parameter
         * @param dataset
         * @param bestGain
         * @param side
         */
        Dataset(Dataset const &dataset, const SplitInfo &bestGain, SplitInfo::Side side):
                _sortedMatrixIdx(dataset.numFeatures()) {

            _X = dataset.X();
            _y = dataset.yptr();

            size_t len, start, end;
            if (side == SplitInfo::Side::Left) {
                start = 0, end = bestGain.getBestSortedIndex();
                len = end;
            } else {
                start = bestGain.getBestSortedIndex(), end = bestGain.size();
                len = end - start;
            }

            _rowIndices = VectorT(len);
            for (size_t i = start ; i < end; i++) {
                _rowIndices[i - start] = dataset._rowIndices[bestGain.getSortedFeatureIndex(i)];
            }

            long cols = dataset.numFeatures();

            for (long j = 0; j < cols; j++) {
                _sortedMatrixIdx[j] = sortIndices(j);
            }
        }

        inline size_t nRows() const {
            return _rowIndices.size();
        }

        inline long numFeatures() const {
            return this->_X->cols();
        }

        inline std::shared_ptr<MatrixType> X() const {
            return _X;
        }

        inline std::shared_ptr<Vector> yptr() const {
            return _y;
        }

        inline Vector y() const {
            Vector proj(_rowIndices.size());
            for (size_t i = 0; i < proj.size(); i++) {
                proj[i] = (*_y)[_rowIndices[i]];
            }
            return proj;
        }

        inline Eigen::RowVectorXd row(long rowIndex) const {
            return _X->row(_rowIndices[rowIndex]);
        }

        /**
         * Sort the sample indices for a given feature index 'feature_id'.
         *
         * It returns sorted indices depending on type of feature (categorical or numeric):
         * Categorical feature: performs mean target encoding (see feature/categorical branch)
         * Numerical feature: natural sort on numeric value
         *
         * @param colIndex Feature / column of above matrix
         */
        inline const VectorT& sortedColumnIndices(long colIndex) const {
            return _sortedMatrixIdx[colIndex];
        }
    };
}
