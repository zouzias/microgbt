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
    * Represents a machine learning "design matrix" and target vector, (X, y)
    * where the rows and columns of matrix X represent the samples and features, respectively. y is the target vector
    * to be predicted
    */
    class Dataset {
    private:

        /**
         * Design matrix, each row correspond to a sample; each column corresponds to a feature
         */
        std::shared_ptr<MatrixType> _X;

        /**
         * Target vector
         */
        std::shared_ptr<Vector> _y;

        VectorT _rowIndices;

        /**
         * Return sorted indices from an Eigen vector
         * @param v
         * @return
         */
         Eigen::VectorXi sortIndices(long colIndex) const{

            // initialize original index locations
            Eigen::VectorXd v = col(colIndex);
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

        Dataset(const MatrixType &X, Vector &y) : _X(std::make_shared<MatrixType>(X)),
        _y(std::make_shared<Vector>(y)),
        _rowIndices(y.size()){
            // By default, all rows are included in the dataset
            std::iota(_rowIndices.begin(), _rowIndices.end(), 0);
        }


        Dataset(Dataset const &dataset) = default;

        /**
         * Construct a Dataset, given a binary split gain and lef/right side parameter
         * @param dataset
         * @param bestGain
         * @param side
         */
        Dataset(Dataset const &dataset, const SplitInfo &bestGain, SplitInfo::Side side) {
            if (side == SplitInfo::Side::Left) {
                _rowIndices = bestGain.getLeftIds();
            } else {
                _rowIndices = bestGain.getRightIds();
            }

            _X = dataset.X();
            _y = dataset.yptr();
        }

        inline size_t nRows() const {
            return this->_rowIndices.size();
        }

        inline VectorT rowIter() const {
            return _rowIndices;
        }

        inline size_t numFeatures() const {
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

        inline Eigen::RowVectorXd col(long colIndex) const {
            Eigen::RowVectorXd column(_rowIndices.size());
            auto fullColumn = _X->col(colIndex);
            for (size_t i = 0; i < _rowIndices.size(); i++) {
                column[i] = fullColumn[_rowIndices[i]];
            }
            return column;
        }

        /**
         * Returns a sorted vector of indices corresponding to a column
         * @param colIndex Index of column
         * @return
         */
        inline Eigen::RowVectorXi sortedColumnIndices(long colIndex) const {
            return sortIndices(colIndex);
        }
    };
}
