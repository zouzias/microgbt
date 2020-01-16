#pragma once
#include "splitter.h"

namespace microgbt {

    /**
     * Splitter on numerical features
     */
    class NumericalSplitter: public Splitter {

        // Regularization parameter of xgboost
        double _lambda;

        /**
        * Returns objective value for a given gradient, hessian and lambda value
        *
        * @param gradient Gradient value
        * @param hessian Hessian value
        * @return
        */
        constexpr double objective(double gradient, double hessian) const {
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
        constexpr double calc_split_gain(double G, double H, double G_l, double H_l) const {
            return objective(G_l, H_l) + objective(G - G_l, H - H_l) - objective(G, H) / 2.0; // TODO: minus \gamma
        }

        /**
        * Returns an optimal binary split for a given feature index of a Dataset.
        *
        * @param dataset Input dataset
        * @param previousPreds
        * @param gradient Gradient vector
        * @param hessian Hessian vector
        * @param featureId Feature index
        * @return Best split over all possible splits of feature with featureId
        */
        SplitInfo optimumGainByFeature(const Dataset &dataset,
                                       const Vector &gradient,
                                       const Vector &hessian,
                                       long featureId) const {

            // Sort the feature by value and return permutation of indices (i.e., argsort)
            const Eigen::RowVectorXi& sortedInstanceIds = dataset.sortedColumnIndices(featureId);

            // Cummulative sum of gradients and Hessian
            Vector cum_sum_G(dataset.nRows()), cum_sum_H(dataset.nRows());
            double cum_sum_g = 0.0, cum_sum_h = 0.0;
            for (size_t i = 0 ; i < dataset.nRows(); i++) {
                long idx = sortedInstanceIds[i];
                cum_sum_g += gradient[idx];
                cum_sum_h += hessian[idx];
                cum_sum_G[i] = cum_sum_g;
                cum_sum_H[i] = cum_sum_h;
            }

            // For each feature, compute split gain and keep the split index with maximum gain
            Vector gainPerOrderedSampleIndex(dataset.nRows());
            for (size_t i = 0 ; i < dataset.nRows(); i++){
                gainPerOrderedSampleIndex[i] = calc_split_gain(cum_sum_g, cum_sum_h, cum_sum_G[i], cum_sum_H[i]);
            }

            long bestGainIndex =
                    std::max_element(gainPerOrderedSampleIndex.begin(), gainPerOrderedSampleIndex.end())
                    - gainPerOrderedSampleIndex.begin();
            double bestGain = gainPerOrderedSampleIndex[bestGainIndex];
            double bestSplitNumericValue = dataset.row(sortedInstanceIds[bestGainIndex])[featureId];
            long bestSortedIndex = bestGainIndex + 1;

            return SplitInfo(sortedInstanceIds, bestGain, bestSplitNumericValue, bestSortedIndex);
        }

    public:
        explicit NumericalSplitter(double lambda) {
            _lambda = lambda;
        }

        SplitInfo findBestSplit(const Dataset &trainSet,
                                const Vector &gradient,
                                const Vector &hessian) const override {

            long numFeatures = trainSet.numFeatures();

            // 1) For each tree node, enumerate over all features:
            // 2) For each feature, sorted the instances by feature numeric value
            //    - Compute gain for every feature (column of design matrix)
            std::vector<SplitInfo> gainPerFeature(numFeatures);
            for (long featureId = 0; featureId < numFeatures; featureId++) {
                gainPerFeature[featureId] = optimumGainByFeature(trainSet, gradient, hessian, featureId);
            }

            // 3) Use a linear scan to decide the best split along that feature
            // 4) Take the best split solution (that maximises gain reduction) over all features
            long bestFeatureId =
                    std::max_element(gainPerFeature.begin(), gainPerFeature.end()) - gainPerFeature.begin();
            SplitInfo bestSplitInfo = gainPerFeature[bestFeatureId];

            bestSplitInfo.setBestFeatureId(bestFeatureId);

            return bestSplitInfo;
        }
    };
} // Namespace microgbt