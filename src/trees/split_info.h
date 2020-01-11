#pragma once
#include <utility>
#include <Eigen/Dense>
#include <vector>

#include "../types.h"

namespace microgbt
{

/**
         * SplitInfo contains information of a binary tree split such as
         * gain value, split numeric value on which best split gain is attained.
         */
class SplitInfo
{

    // Sorted list of feature indices with respect to feature values
    VectorT _leftSplit, _rightSplit;

    // Sorted list of feature indices with respect to feature values
    Eigen::RowVectorXi _sortedFeatureIndices;

    /* Best gain of split and split value on which the best gain is attained */
    double _bestGain = std::numeric_limits<double>::min(), _bestSplitNumericValue = 0.0;

    // Feature index on which best gain was attained
    long _bestFeatureId = -1;

public:
    enum Side{ Left, Right};

    explicit SplitInfo() = default;

    SplitInfo(double gain, double bestSplitNumericValue)
    {
        _bestGain = gain;
        _bestSplitNumericValue = bestSplitNumericValue;
    }

    SplitInfo(double gain, double bestSplitNumericValue, VectorT leftSplit, VectorT rightSplit):
        _leftSplit(std::move(leftSplit)), _rightSplit(std::move(rightSplit)){
        _bestGain = gain;
        _bestSplitNumericValue = bestSplitNumericValue;
    }

    bool operator<(const SplitInfo &rhs) const { return this->_bestGain <= rhs.bestGain(); }

    inline double bestGain() const { return _bestGain; }

    inline double splitValue() const { return _bestSplitNumericValue; }

    inline double getBestFeatureId() const { return _bestFeatureId; }

    void setBestFeatureId(size_t bestFeatureId) { _bestFeatureId = bestFeatureId; }

    std::shared_ptr<VectorT> getLeftLocalIds() const { return std::make_shared<VectorT>(_leftSplit); }

    std::shared_ptr<VectorT>  getRightLocalIds() const { return std::make_shared<VectorT>(_rightSplit); }

        /**
         * Split a vector based on a side, i.e., left and right side.
         *
         * SplitInfo has the left and right subset of indices corresponding to the left and right subtree, respectively.
         *
         * @param vec Input vector to split
         * @param side Left or right side
         * @return a sub-vector of the input vector based on the split information
         */
        VectorD split(const VectorD &vec, const SplitInfo::Side &side) const {
            VectorD splitVector;
            std::shared_ptr<VectorT> rowIndices;
            if (side == SplitInfo::Side::Left) {
                rowIndices = getLeftLocalIds();
            } else {
                rowIndices = getRightLocalIds();
            }

            splitVector.reserve(rowIndices->size());
            for (size_t i = 0; i < rowIndices->size(); i++){
                splitVector.push_back(vec[(*rowIndices)[i]]);
            }

            return splitVector;
        }
    };
}