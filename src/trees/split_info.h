#pragma once
#include <utility>
#include <Eigen/Dense>
#include<vector>


namespace microgbt {

        using VectorD = std::vector<double>;
        using VectorT = std::vector<size_t>;

        /**
         * SplitInfo contains information of a binary tree split such as
         * gain value, split numeric value on which best split gain is attained.
         */
        class SplitInfo {

            // Sorted list of feature indices with respect to feature values
            Eigen::RowVectorXi _sortedFeatureIndices;

            /* Best gain of split and split value on which the best gain is attained */
            double _bestGain = std::numeric_limits<double>::min(), _bestSplitNumericValue = 0.0;

            // Keep track of best sorted sample index
            size_t _bestSortedIndex = -1;

            // Feature index on which best gain was attained
            size_t _bestFeatureId = -1;

            public:

            enum Side{
                Left,
                Right
            };

            explicit SplitInfo() = default;

            SplitInfo(double gain, double bestSplitNumericValue) {
                _bestGain = gain;
                _bestSplitNumericValue = bestSplitNumericValue;
            }

            SplitInfo(Eigen::RowVectorXi  sortedFeatureIndices, double gain, double bestSplitNumericValue, size_t bestSortedIdx):
                _sortedFeatureIndices(std::move(sortedFeatureIndices)) {
                _bestGain = gain;
                _bestSplitNumericValue = bestSplitNumericValue;
                _bestSortedIndex = bestSortedIdx;
            }

            bool operator < (const SplitInfo& rhs) const { return this->_bestGain <= rhs.bestGain(); }

            inline double bestGain() const {
                return _bestGain;
            }

            inline double splitValue() const {
                return _bestSplitNumericValue;
            }

            void setBestFeatureId(size_t bestFeatureId) {
                _bestFeatureId = bestFeatureId;
            }
            inline size_t getBestFeatureId() const {
                return _bestFeatureId;
            }

            VectorT getLeftLocalIds() const {
                return VectorT(_sortedFeatureIndices.data(),
                        _sortedFeatureIndices.data() + _bestSortedIndex);
            }

            VectorT getRightLocalIds() const {
                return VectorT(_sortedFeatureIndices.data() + _bestSortedIndex,
                        _sortedFeatureIndices.data() + _sortedFeatureIndices.size());
            }

            /**
             * Split a vector based on a side, i.e., left and right side.
             *
             * SplitInfo has the left and right subset of indices corresponding to the left and right subtree, respectively.
             *
             * @param vector Input vector to split
             * @param side Left or right side
             * @return a sub-vector of the input vector based on the split information
             */
            VectorD split(const VectorD &vector, const SplitInfo::Side &side) const {
                VectorT rowIndices;
                if (side == SplitInfo::Side::Left) {
                    rowIndices = getLeftLocalIds();
                } else {
                    rowIndices = getRightLocalIds();
                }

                VectorD splitVector;
                std::transform(rowIndices.begin(), rowIndices.end(),
                        std::back_inserter(splitVector),
                        [&vector](size_t rowIndex){
                    return vector[rowIndex];
                });

                return splitVector;
            }
        };
} // namespace microgbt
