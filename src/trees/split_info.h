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
            VectorT _sortedFeatureIndices;

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

            SplitInfo(const VectorT& sortedFeatureIndices, double gain, double bestSplitNumericValue, size_t bestSortedIdx):
                _sortedFeatureIndices(sortedFeatureIndices) {
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

            inline size_t getBestSortedIndex() const {
                return _bestSortedIndex;
            }

            inline size_t getSortedFeatureIndex(long i) const {
                return _sortedFeatureIndices[i];
            }

            inline size_t size() const {
                return _sortedFeatureIndices.size();
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
                size_t len, start, end;
                if (side == SplitInfo::Side::Left) {
                    start = 0, end = _bestSortedIndex;
                    len = _bestSortedIndex;
                } else {
                    start = _bestSortedIndex, end = _sortedFeatureIndices.size();
                    len = _sortedFeatureIndices.size() - _bestSortedIndex;
                }

                VectorD splitVector(len);
                for( size_t i = start ; i < end; i++){
                    splitVector[i - start] = vector[_sortedFeatureIndices[i]];
                }

                return splitVector;
            }
        };
} // namespace microgbt
