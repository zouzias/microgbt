#pragma once
#include<vector>


namespace microgbt {

        using VectorD = std::vector<double>;
        using VectorT = std::vector<size_t>;

        /**
         * SplitInfo contains information of a binary tree split such as
         * gain value, split numeric value on which best split gain is attained.
         */
        class SplitInfo {

            private:

            /* Best gain and value (!?) */
            double _bestGain = std::numeric_limits<double>::min(), _bestSplitNumericValue = 0;

            /**
             * List of sample indices of the left subtree on an optimal binary tree split
             */
            VectorT _bestLeftInstanceIds;

            /**
             * List of sample indices of the right subtree on an optimal binary tree split
             */
            VectorT _bestRightInstanceIds;


            VectorT _bestLeftLocalIds;

            VectorT _bestRightLocalIds;

            public:

            enum Side{
                Left,
                Right
            };

            explicit SplitInfo() = default;

            SplitInfo(double
            gain, double
            value) {
                _bestGain = gain;
                _bestSplitNumericValue = value;
            }

            SplitInfo(double
                 gain, double
                 value,
                 VectorT& bestLeft,
                 VectorT& bestRight,
                 VectorT& bestLocalLeft,
                 VectorT& bestLocalRight): _bestLeftInstanceIds(bestLeft),_bestRightInstanceIds(bestRight),
                      _bestLeftLocalIds(bestLocalLeft), _bestRightLocalIds(bestLocalRight){
                _bestGain = gain;
                _bestSplitNumericValue = value;
            }

            bool operator < (const SplitInfo& rhs) const { return this->_bestGain <= rhs.bestGain(); }

            double bestGain() const {
                return _bestGain;
            }

            double splitValue() const {
                return _bestSplitNumericValue;
            }

            VectorT getLeftIds() const {
                return _bestLeftInstanceIds;
            }

            VectorT getRightIds() const {
                return _bestRightInstanceIds;
            }

            VectorD split(const VectorD &vector, const SplitInfo::Side &side) const {
                VectorT rowIndices;
                if (side == SplitInfo::Side::Left)
                    rowIndices = _bestLeftLocalIds;
                else
                    rowIndices = _bestRightLocalIds;

                VectorD splitVector;
                std::transform(rowIndices.begin(), rowIndices.end(),
                        std::back_inserter(splitVector),
                        [&vector](size_t rowIndex){
                    return vector[rowIndex];
                });

                return splitVector;
            }
        };
}