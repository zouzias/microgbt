#pragma once
#include <vector>
#include<iostream>
#include <Eigen/Dense>
#include "trees/split_info.h"

namespace microgbt {

    using Vector = std::vector<double>;

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
        Eigen::MatrixXd _X;

        /**
         * Target vector
         */
        Vector _y;
    public:

        Dataset() = default;

        Dataset(const Eigen::MatrixXd& X, const Vector& y): _X(X), _y(y) {
        }


        Dataset(Dataset const &dataset) = default;

        /**
         * Construct a Dataset, given a binary split gain and lef/right side parameter
         * @param dataset
         * @param bestGain
         * @param side
         */
        Dataset(Dataset const &dataset, const SplitInfo& bestGain, SplitInfo::Side side) {

            std::vector<size_t> rowIndices;
            if (side == SplitInfo::Side::Left) {
                rowIndices = bestGain.getLeftIds();
            }
            else {
                rowIndices = bestGain.getRightIds();
            }

            int rowSize = rowIndices.size(), colSize = dataset.numFeatures();

            _y = Vector(dataset.y());
            _X = Eigen::MatrixXd(rowSize, colSize);

            int i = 0;
            for (auto idx: rowIndices) {
                for (int j = 0 ; j < colSize; j++) {
                    _X(i, j) = dataset.X()(idx, j);
                }
                i++;
            }
        }

        inline size_t nRows() const {
            return this->_X.rows();
        }

        inline size_t numFeatures() const {
            return this->_X.cols();
        }

        Vector y() const {
            return _y;
        }

        Eigen::MatrixXd X() const {
            return _X;
        }

        Eigen::RowVectorXd row(long rowIndex) const {
            return _X.row(rowIndex);
        }

        Eigen::RowVectorXd col(long colIndex) const {
            return _X.col(colIndex);
        }
    };
}