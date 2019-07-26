#pragma once
#include <vector>
#include<iostream>
#include <Eigen/Dense>
#include <numeric>
#include "trees/split_info.h"

namespace microgbt {

    using Vector = std::vector<double>;
    using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using SortedMatrixType = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    /**
    * Represents a machine learning "design matrix" and target vector, (X, y)
    * where the rows and columns of matrix X represent the samples and features, respectively. y is the target vector
    * to be predicted
    */
    class Dataset {
    private:

        /**
         * Design matrix, each row correspond to a sample; each column corresponds to a feature
         */
        MatrixType _X;


        /**
         * Matrix whose columns contain the
         */
        SortedMatrixType _sortedColumnValues;

        /**
         * Target vector
         */
        Vector _y;

        /**
         * Return sorted indices from an Eigen vector
         * @param v
         * @return
         */
         Eigen::VectorXi sortIndices(int colIndex) const{

            // initialize original index locations
            Eigen::VectorXd v = _X.col(colIndex);
            unsigned int n = v.size();

            Eigen::VectorXi idx(n);
            // idx contains now 0,1,...,v.size() - 1
            std::iota(idx.data(), idx.data() + idx.size(), 0);

            // sort indexes based on comparing values in v
            std::sort(idx.data(), idx.data() + idx.size(),
                 [&v](int i1, int i2) {return v[i1] < v[i2];});

            return idx;
        }

    public:

        Dataset() = default;

        Dataset(const MatrixType &X, Vector &y) : _X(X), _sortedColumnValues(X.rows(), X.cols()), _y(y) {

            // Compute sorted indices per column
            for (long j = 0; j < X.cols(); j++) {
                _sortedColumnValues.col(j) = sortIndices(j);
            }
        }


        Dataset(Dataset const &dataset) = default;

        /**
         * Construct a Dataset, given a binary split gain and lef/right side parameter
         * @param dataset
         * @param bestGain
         * @param side
         */
        Dataset(Dataset const &dataset, const SplitInfo &bestGain, SplitInfo::Side side): _y(dataset.y()) {

            std::vector<size_t> rowIndices;
            if (side == SplitInfo::Side::Left) {
                rowIndices = bestGain.getLeftIds();
            } else {
                rowIndices = bestGain.getRightIds();
            }

            int rowSize = rowIndices.size(), colSize = dataset.numFeatures();

            _X = MatrixType(rowSize, colSize);
            _sortedColumnValues = SortedMatrixType(rowSize, _X.cols());

            int i = 0;
            for (auto idx: rowIndices) {
                _X.row(i) = dataset.row(idx);
                i++;
            }

            // Compute sorted indices per column
            for (int j = 0; j < colSize; j++) {
                _sortedColumnValues.col(j) = sortIndices(j);
            }
        }

        inline size_t nRows() const {
            return this->_X.rows();
        }

        inline size_t numFeatures() const {
            return this->_X.cols();
        }

        inline Vector y() const {
            return _y;
        }

        inline Eigen::RowVectorXd row(long rowIndex) const {
            return _X.row(rowIndex);
        }

        /**
         * Returns a sorted vector of indices corresponding to a column
         * @param colIndex Index of column
         * @return
         */
        inline Eigen::RowVectorXi sortedColumnIndices(long colIndex) const {
            return _sortedColumnValues.col(colIndex);
        }
    };
}