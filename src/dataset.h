#pragma once
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <numeric>
#include <memory>

#include "trees/split_info.h"
#include "types.h"

namespace microgbt
{

/**
    * Dataset represents a machine learning "design matrix" and target vector, (X, y)
    * where the rows and columns of matrix X represent the samples and features, respectively. y is the target vector
    * to be predicted
    */
class Dataset
{

    // Design matrix, each row corresponds to a sample; each column corresponds to a feature
    std::shared_ptr<MatrixType> _X;

    // Target vector
    std::shared_ptr<Vector> _y;

    SortedMatrixType _sortedMatrixIdx;

    VectorT _rowIndices;

    /**
         * Return sorted indices from an Eigen vector
         * @param v
         * @return
         */
    Eigen::VectorXi sortIndices(long colIndex) const
    {

        // initialize original index locations
        Eigen::VectorXd v = col(colIndex);
        long n = v.size();

        Eigen::VectorXi idx(n);
        // idx contains now 0,1,...,v.size() - 1
        std::iota(idx.data(), idx.data() + idx.size(), 0);

        // sort indexes based on comparing values in v
        std::sort(idx.data(), idx.data() + idx.size(),
                  [&v](long i1, long i2) { return v[i1] < v[i2]; });

        return idx;
    }

    inline Eigen::RowVectorXd col(long colIndex) const
    {
        Eigen::RowVectorXd column(_rowIndices.size());
        for (size_t i = 0; i < _rowIndices.size(); i++)
        {
            column[i] = _X->coeffRef(_rowIndices[i], colIndex);
        }
        return column;
    }

public:
    Dataset() = default;

    Dataset(const MatrixType &X, const Vector &y) : _sortedMatrixIdx(X.rows(), X.cols()),
                                                    _rowIndices(y.size())
    {
        _X = std::make_shared<MatrixType>(X);
        _y = std::make_shared<Vector>(y);
        // By default, all rows are included in the dataset
        std::iota(_rowIndices.begin(), _rowIndices.end(), 0);

        for (long j = 0; j < X.cols(); j++)
        {
            _sortedMatrixIdx.col(j) = sortIndices(j);
        }
    }

    Dataset(Dataset const &dataset) = default;

    /**
         * Construct a Dataset, given a binary split gain and lef/right side parameter
         * @param dataset
         * @param bestGain
         * @param side
         */
    Dataset(Dataset const &dataset, const SplitInfo &bestGain, SplitInfo::Side side)
    {

        _X = dataset.X();
        _y = dataset.yptr();

        VectorT localIds;
        if (side == SplitInfo::Side::Left)
        {
            localIds = bestGain.getLeftLocalIds();
        }
        else
        {
            localIds = bestGain.getRightLocalIds();
        }

        _rowIndices = VectorT(localIds.size());
        VectorT otherRowIndices = dataset.rowIter();
        for (size_t i = 0; i < localIds.size(); i++)
        {
            _rowIndices[i] = otherRowIndices[localIds[i]];
        }

        size_t rows = _rowIndices.size();
        long cols = dataset.numFeatures();

        _sortedMatrixIdx = SortedMatrixType(rows, cols);

        for (long j = 0; j < cols; j++)
        {
            _sortedMatrixIdx.col(j) = sortIndices(j);
        }
    }

    inline long nRows() const { return static_cast<long>(this->_rowIndices.size()); }

    inline VectorT rowIter() const { return _rowIndices; }

    inline long numFeatures() const { return this->_X->cols(); }

    inline std::shared_ptr<MatrixType> X() const { return _X; }

    inline std::shared_ptr<Vector> yptr() const { return _y; }

    inline Vector y() const
    {
        Vector proj(_rowIndices.size());
        for (size_t i = 0; i < proj.size(); i++)
        {
            proj[i] = (*_y)[_rowIndices[i]];
        }
        return proj;
    }

    inline Eigen::RowVectorXd row(long rowIndex) const { return _X->row(_rowIndices[rowIndex]); }

    /**
         * Sort the sample indices for a given feature index 'feature_id'.
         *
         * It returns sorted indices depending on type of feature (categorical or numeric):
         * Categorical feature: performs mean target encoding (see feature/categorical branch)
         * Numerical feature: natural sort on numeric value
         *
         * @param colIndex Feature / column of above matrix
         */
    inline Eigen::RowVectorXi sortedColumnIndices(long colIndex) const { return _sortedMatrixIdx.col(colIndex); }
};
} // namespace microgbt
