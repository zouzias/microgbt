#pragma once
#include <utility>

#include "splitter.h"

namespace microgbt
{

/**
     * Splitter on numerical features
     */
class NumericalSplitter : public Splitter
{

    std::vector<Histogram> _histograms;

    // Regularization parameter of xgboost
    double _lambda;

    /**
        * Returns objective value for a given gradient, hessian and lambda value
        *
        * @param gradient Gradient value
        * @param hessian Hessian value
        * @return
        */
    constexpr double objective(double gradient, double hessian) const
    {
        return (gradient * gradient) / (hessian + _lambda);
    }

    /**
        * Returns gain difference of a specific binary tree split.
        *
        * Refer to Eq7 of Reference [1]
        *
        * @param G Gradient on node before the split applied
        * @param H Hessian on node before the split applied
        * @param G_l Gradient on left split node
        * @param H_l Hesssian on left split node
        * @return Gain on split, i.e., reduction on objective value
        */
    constexpr double calc_split_gain(double G, double H, double G_l, double H_l) const
    {
        return objective(G_l, H_l) + objective(G - G_l, H - H_l) - objective(G, H) / 2.0; // TODO: minus \gamma
    }

    /**
        * Returns an optimal binary split for a given feature index of a Dataset.
        *
        * @param dataset Input dataset
        * @param featureId Feature index
        * @return Best split over all possible splits of feature 'featureId'
        */
        SplitInfo optimumGainByFeature(const Dataset& dataset, long featureId) const {

            // Cummulative sum of gradients and Hessian
            const Histogram* histogram = dataset.histogram(featureId);
            long cum_sum_total_size = 0;
            long numBins = histogram->numBins();
            VectorD cum_sum_G(numBins), cum_sum_H(numBins);
            VectorT cum_sum_size(numBins);
            double cum_sum_g = 0.0, cum_sum_h = 0.0;
            for (long i = 0 ; i < numBins; i++) {
                cum_sum_g += histogram->gradientAtBin(i);
                cum_sum_h += histogram->hessianAtBin(i);
                cum_sum_total_size += histogram->getCount(i);
                cum_sum_G[i] = cum_sum_g;
                cum_sum_H[i] = cum_sum_h;
                cum_sum_size[i] = cum_sum_total_size;
            }

            // For each feature, compute split gain and keep the split index with maximum gain
            VectorD gainPerOrderedSampleIndex(numBins);
            for (long i = 0 ; i < numBins; i++){
                gainPerOrderedSampleIndex[i] = calc_split_gain(cum_sum_g, cum_sum_h, cum_sum_G[i], cum_sum_H[i]);
            }

            long bestGainIndex =
                    std::max_element(gainPerOrderedSampleIndex.begin(), gainPerOrderedSampleIndex.end())
                    - gainPerOrderedSampleIndex.begin();
            double bestGain = gainPerOrderedSampleIndex[bestGainIndex];
            double bestSplitNumericValue = histogram->upperThreshold(bestGainIndex);

            VectorT leftSplit, rightSplit;
            leftSplit.reserve(cum_sum_size[bestGainIndex]);
            rightSplit.reserve(dataset.nRows() - cum_sum_size[bestGainIndex]);
            for (long i = 0; i < dataset.nRows(); i++) {
                if ( dataset.coeff(i, featureId) < bestSplitNumericValue) {
                    leftSplit.push_back(i);
                } else {
                    rightSplit.push_back(i);
                }
            }


            return SplitInfo(bestGain, bestSplitNumericValue, leftSplit, rightSplit);
        }

    public:

        explicit NumericalSplitter(double lambda): _lambda(lambda) {}


        SplitInfo findBestSplit(const Dataset& dataset) const override {
            size_t numFeatures = dataset.numFeatures();

            // 1) For each tree node, enumerate over all features:
            // 2) For each feature, sorted the instances by feature numeric value
            //    - Compute gain for every feature (column of design matrix)
            std::vector<SplitInfo> gainPerFeature(numFeatures);
            for (size_t featureId = 0; featureId < numFeatures; featureId++) {
                gainPerFeature[featureId] = optimumGainByFeature(dataset, featureId);
            }

            // 3) Use a linear scan to decide the best split along that feature
            // 4) Take the best split solution (that maximises gain reduction) over all features
            size_t bestFeatureId =
                    std::max_element(gainPerFeature.begin(), gainPerFeature.end()) - gainPerFeature.begin();
            SplitInfo bestSplitInfo = gainPerFeature[bestFeatureId];

        bestSplitInfo.setBestFeatureId(bestFeatureId);

        return bestSplitInfo;
    }
};
} // Namespace microgbt