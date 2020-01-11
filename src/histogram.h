#pragma once

#include <cmath>
#include "dataset.h"

namespace microgbt {

    using VectorD = std::vector<double>;

    /**
     * A Gradient Boosting related Histogram
     *
     * Given a tuple of ("feature vector", "gradient vector", "hessian vector") constructs a histogram with domain
     * the feature vector values and binned gradient / hessian values
     */
    class Histogram {

        double SMALLEST_BIN_LENGTH = 10e-6;

        // Number of bins / samples
        long _numBins = 0, _numSamples = 0;

        // Histogram values on Gradients / Hessians
        std::vector<double> _gradientHist,  _hessianHist;

        // Histogram counts
        std::vector<long> _count;

        // Histogram limits
        double _minValue = std::numeric_limits<double>::max(), _maxValue = std::numeric_limits<double>::min(), _binLength = 0;

    public:

        Histogram() = default;

        Histogram(const VectorD &featureValues, const VectorD &gradients, const VectorD &hessians, size_t numBins) :
                _numBins(numBins), _gradientHist(numBins), _hessianHist(numBins), _count(numBins) {

            _numSamples = gradients.size();

            // Setup the regular axis (min, max, numBins)
            for (auto v: featureValues) {
                if ( v < _minValue) {
                    _minValue = v;
                }

                if ( v > _maxValue) {
                    _maxValue = v;
                }
            }

            // Length of bins
            _binLength = std::max((_maxValue - _minValue) / (double)_numBins, SMALLEST_BIN_LENGTH);

            // Fill in the histogram here
            for (long i = 0; i < _numSamples; i++) {
                long binIndex = bin(featureValues[i]);
                _gradientHist[binIndex] += gradients[i];
                _hessianHist[binIndex] += hessians[i];
                _count[binIndex]++;
            }
        }

        void fillValues(const VectorD& featureValues, const VectorD& gradients, const VectorD& hessians) {
            _numSamples = featureValues.size();
            std::fill(_gradientHist.begin(), _gradientHist.end(), 0.0);
            std::fill(_hessianHist.begin(), _hessianHist.end(), 0.0);
            std::fill(_count.begin(), _count.end(), 0);

            // Fill in the histogram here
            for (long i = 0; i < _numSamples; i++) {
                long binIndex = bin(featureValues[i]);
                _gradientHist[binIndex] += gradients[i];
                _hessianHist[binIndex] += hessians[i];
                _count[binIndex]++;
            }
        }

        double upperThreshold(long binIndex) const {
            if (binIndex == _numBins - 1) { return std::numeric_limits<double>::max(); }
            return _minValue + (double) (binIndex + 1) * _binLength;
        }

        double lowerThreshold(long binIndex) const {
            if (binIndex == 0) { return std::numeric_limits<double>::min(); }
            if (binIndex == _numBins - 1) { return _maxValue; }

            return _minValue + (double) binIndex * _binLength;
        }

        long numBins() const { return _numBins; }

        double binLength() const { return _binLength; }

        double gradientAtBin(long binIndex) const { return _gradientHist[binIndex]; }

        double hessianAtBin(long binIndex) const { return _hessianHist[binIndex]; }

        long getCount(long binIndex) const { return _count[binIndex]; }

        double min() const { return _minValue; }

        double max() const { return _maxValue; }

        /**
         * Returns index of histogram bin that input value is mapped.
         *
         * Underflow values go to first bin (index 0) and overflow values go to latest bin (index _numBins - 1)
         *
         * @param value
         * @return Index of histogram bin that input value is mapped
         */
        inline long bin(double value) const {
            if ( value < _minValue) { return 0; }
            if ( value > _maxValue) { return _numBins - 1; }

            return std::min(static_cast<long>(std::floor( (value - _minValue) / _binLength)), _numBins - 1);
        }
    };
}