#pragma once
#include "splitter.h"
#include "../utils.h"


namespace microgbt {

    /**
     * Splitter on numerical features
     */
    class NumericalSplitter: public Splitter {

        double _lambda;

        /**
       * Sort the sample indices for a given feature index 'feature_id'.
       *
       * It returns sorted indices depending on type of feature (categorical or numeric):
       * Categorical feature: performs mean target encoding
       * Numerical feature: natural sort on numeric value
       *
       * @param trainSet Input design matrix and targets as Dataset
       * @param featureId Feature / column of above matrix
       */
        static Eigen::RowVectorXi sortSamplesByFeature(const Dataset &trainSet,
                                                       int featureId) {

            return trainSet.sortedColumnIndices(featureId);
        }

        /**
        * Returns objective value for a given gradient, hessian and lambda value
        *
        * @param gradient
        * @param hessian
        * @param lambd
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
        * @param G_r Gradient on right split node
        * @param H_r Hesisan on right split node
        * @param lambd Regularization xgboost parameter, see Eqn. 7 in [1]
        * @return
        */
        constexpr double calc_split_gain(double G, double H, double G_l, double H_l) const {
            return objective(G_l, H_l) + objective(G - G_l, H - H_l) - objective(G, H) / 2.0; // TODO: minus \gamma
        }

        /**
        * Returns an optimal binary split for a given feature index of a Dataset.
        *
        * @param dataset
        * @param previousPreds
        * @param gradient
        * @param hessian
        * @param featureId
        * @return
        */
        SplitInfo optimumGainByFeature(const Dataset &dataset,
                                       const Vector &gradient,
                                       const Vector &hessian,
                                       int featureId) const {

            double G = par_simd_accumulate(gradient);
            double H = par_simd_accumulate(hessian);

            // Sort the feature by value and return permutation of indices (i.e., argsort)
            Eigen::RowVectorXi sortedInstanceIds = sortSamplesByFeature(dataset, featureId);

            // Cummulative sum of gradients and Hessian
            Vector cum_sum_G(dataset.nRows());
            Vector cum_sum_H(dataset.nRows());
            double cum_sum_g = 0.0, cum_sum_h = 0.0;
            for (size_t i = 0 ; i < dataset.nRows(); i++) {
                cum_sum_g += gradient[sortedInstanceIds[i]];
                cum_sum_h += hessian[sortedInstanceIds[i]];
                cum_sum_G[i] = cum_sum_g;
                cum_sum_H[i] = cum_sum_h;
            }

            // For each feature, compute split gain and keep the split index with maximum gain
            Vector gainPerOrderedSampleIndex(dataset.nRows());
            for (size_t i = 0 ; i < dataset.nRows(); i++){
                gainPerOrderedSampleIndex[i] = calc_split_gain(G, H, cum_sum_G[i], cum_sum_H[i]);
            }

            long bestGainIndex =
                    std::max_element(gainPerOrderedSampleIndex.begin(), gainPerOrderedSampleIndex.end())
                    - gainPerOrderedSampleIndex.begin();
            double bestGain = gainPerOrderedSampleIndex[bestGainIndex];
            double bestSplitNumericValue = dataset.row(sortedInstanceIds[bestGainIndex])[featureId];
            size_t bestSortedIndex = bestGainIndex + 1;

            return SplitInfo(sortedInstanceIds, bestGain, bestSplitNumericValue, bestSortedIndex);
        }



    public:
        explicit NumericalSplitter(double lambda) {
            _lambda = lambda;
        }


        SplitInfo findBestSplit(const Dataset &trainSet,
                                const Vector &gradient,
                                const Vector &hessian) const override {

            size_t numFeatures = trainSet.numFeatures();

            // 1) For each tree node, enumerate over all features:
            // 2) For each feature, sorted the instances by feature numeric value
            //    - Compute gain for every feature (column of design matrix)
            std::vector<SplitInfo> gainPerFeature(numFeatures);
            for (size_t featureId = 0; featureId < numFeatures; featureId++) {
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