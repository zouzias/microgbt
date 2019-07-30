#pragma once
#include<memory>
#include<map>
#include<vector>
#include<string>
#include<algorithm>
#include<numeric>
#include<iterator>
#include<iostream>
#include <thread>

#include "../dataset.h"
#include "split_info.h"
#include "../utils.h"


namespace microgbt {

    /**
     * A node of a tree
     */
    class TreeNode {
    private:
        int _maxDepth;
        double _lambda, _minSplitGain, _minTreeSize;
        bool isLeaf = false;
        std::unique_ptr<TreeNode> leftSubTree;
        std::unique_ptr<TreeNode> rightSubTree;
        long splitFeatureIndex;

        /**
         * Numeric value on which a binary tree split took place
         */
        double splitNumericValue;
        double weight = 0.0;

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

    public:

        TreeNode(double lambda, double minSplitGain, double minTreeSize, int maxDepth){
            _lambda = lambda;
            _minSplitGain = minSplitGain;
            _maxDepth = maxDepth;
            _minTreeSize = minTreeSize;
        }

        /**
         * Returns objective value for a given gradient, hessian and lambda value
         *
         * @param gradient
         * @param hessian
         * @param lambd
         * @return
         */
        inline double objective(double gradient, double hessian) const {
            return pow(gradient, 2) / (hessian + _lambda);
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
        inline double calc_split_gain(double G, double H, double G_l, double H_l) const {
            double G_r = G - G_l;
            double H_r = H - H_l;
            return objective(G_l, H_l) + objective(G_r, H_r) - objective(G, H) / 2.0; // TODO: minus \gamma
        }

        /**

         */
         /**
          * Return  the optimal weight of a leaf node.
          * (Refer to Eq5 of Reference[1])
          *
          * @param gradient
          * @param hessian
          * @return
          */
        inline double calc_leaf_weight(const Vector &gradient,
                                       const Vector &hessian) const {
            return par_simd_accumulate(gradient)
                   / (par_simd_accumulate(hessian) + _lambda);
        }

        /**
         * Returns an optimal binary split for a given feature index of a Dataset.
         *
         * @param trainSet
         * @param previousPreds
         * @param gradient
         * @param hessian
         * @param featureId
         * @return
         */
        SplitInfo optimumGainByFeature(const Dataset &trainSet,
                                  const Vector &gradient,
                                  const Vector &hessian,
                                  int featureId) const {

            double G = par_simd_accumulate(gradient);
            double H = par_simd_accumulate(hessian);

            // Sort the feature by value and return permutation of indices (i.e., argsort)
            Eigen::RowVectorXi sortedInstanceIds = sortSamplesByFeature(trainSet, featureId);

            // Cummulative sum of gradients and Hessian
            Vector cum_sum_G(trainSet.nRows());
            Vector cum_sum_H(trainSet.nRows());
            double cum_sum_g = 0.0, cum_sum_h = 0.0;
            for (size_t i = 0 ; i < trainSet.nRows(); i++) {
                cum_sum_g += gradient[sortedInstanceIds[i]];
                cum_sum_h += hessian[sortedInstanceIds[i]];
                cum_sum_G[i] = cum_sum_g;
                cum_sum_H[i] = cum_sum_h;
            }

            // For each feature, compute split gain and keep the split index with maximum gain
            Vector gainPerOrderedSampleIndex(trainSet.nRows());
            for (size_t i = 0 ; i < trainSet.nRows(); i++){
                gainPerOrderedSampleIndex[i] = calc_split_gain(G, H, cum_sum_G[i], cum_sum_H[i]);
            }

            long bestGainIndex =
                    std::max_element(gainPerOrderedSampleIndex.begin(), gainPerOrderedSampleIndex.end())
                    - gainPerOrderedSampleIndex.begin();
            double bestGain = gainPerOrderedSampleIndex[bestGainIndex];
            double bestSplitNumericValue = trainSet.row(sortedInstanceIds[bestGainIndex])[featureId];
            size_t bestSortedIndex = bestGainIndex + 1;

            return SplitInfo(sortedInstanceIds, bestGain, bestSplitNumericValue, bestSortedIndex);
        }


        /**
         * Recursively (and greedily) split a TreeNode based on
         *
         * Exact Greedy Algorithm for Split Finding:
         * 1) For each tree node, enumerate over all features:
         * 2) For each feature, sorted the instances by feature numeric value
         * 3) Use a linear scan to decide the best split along that feature (if categorical perform Mean Target Encoding)
         * 4) Take the best split solution (that maximises gain reduction) over all features
         * 5) Recurse on the left and right side of the best split
         *
         *  (Refer to Algorithm1 of Reference[1])
         *
         * @param trainSet
         * @param previousPreds
         * @param gradient
         * @param hessian
         * @param shrinkage
         * @param depth
         */
        void build(const Dataset &trainSet,
                   const Vector &previousPreds,
                   const Vector &gradient,
                   const Vector &hessian,
                   double shrinkage,
                   int depth) {

            size_t numFeatures = trainSet.numFeatures();

            // Check if depth is reached
            if (depth > _maxDepth) {
                this->isLeaf = true;
                this->weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            // Check if # of sample is too small
            if ( trainSet.nRows() <= _minTreeSize) {
                this->isLeaf = true;
                this->weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

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
            SplitInfo bestGain = gainPerFeature[bestFeatureId];

            // Check if best gain is less than minimum split gain (threshold)
            if (bestGain.bestGain() < this->_minSplitGain) {
                this->isLeaf = true;
                this->weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            this->splitFeatureIndex = bestFeatureId;
            this->splitNumericValue = bestGain.splitValue();


            Dataset leftDataset(trainSet, bestGain, SplitInfo::Side::Left);
            Vector leftGradient = bestGain.split(gradient, SplitInfo::Side::Left);
            Vector leftHessian = bestGain.split(hessian, SplitInfo::Side::Left);
            Vector leftPreviousPreds = bestGain.split(previousPreds, SplitInfo::Side::Left);
            this->leftSubTree = std::unique_ptr<TreeNode>(
                    new TreeNode(_lambda, _minSplitGain, _minTreeSize, _maxDepth));
            leftSubTree->build(leftDataset, leftPreviousPreds, leftGradient, leftHessian, shrinkage, depth + 1);

            Dataset rightDataset(trainSet, bestGain, SplitInfo::Side::Right);
            Vector rightGradient = bestGain.split(gradient, SplitInfo::Side::Right);
            Vector rightHessian = bestGain.split(hessian, SplitInfo::Side::Right);
            Vector rightPreviousPreds = bestGain.split(previousPreds, SplitInfo::Side::Right);

            this->rightSubTree = std::unique_ptr<TreeNode>(
                    new TreeNode(_lambda, _minSplitGain, _minTreeSize, _maxDepth));
            rightSubTree->build(rightDataset, rightPreviousPreds, rightGradient, rightHessian, shrinkage, depth + 1);
        }

        /**
         * Return the score for a given sample, i.e. set of features
         *
         * @param sample
         * @return
         */
        double score(const Eigen::RowVectorXd &sample) const {
            if (this->isLeaf) {
                return this->weight;
            } else if (sample[this->splitFeatureIndex] < this->splitNumericValue) {
                return this->leftSubTree->score(sample);
            } else {
                return this->rightSubTree->score(sample);
            }
        }
    };
} // namespace microgbt
