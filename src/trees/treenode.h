#pragma once
#include<memory>
#include<map>
#include<vector>
#include<string>
#include<algorithm>
#include<numeric>
#include<iterator>
#include<iostream>

#include "../dataset.h"
#include "split_info.h"


namespace microgbt {

    using Vector = std::vector<double>;

    /**
     * A node of a tree
     */
    class TreeNode {
    private:
        int _maxDepth;
        double _lambda, _minSplitGain, _minTreeSize;
        bool isLeaf = false;
        std::unique_ptr<TreeNode> _leftSubTree;
        std::unique_ptr<TreeNode> _rightSubTree;
        long _splitFeatureIndex;

        /**
         * Numeric value on which a binary tree split took place
         */
        double _splitNumericValue;
        double _weight = 0.0;

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
            _splitNumericValue = 0.0;
            _splitFeatureIndex = -1;
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
        inline double calc_split_gain(double G, double H, double G_l, double H_l, double G_r, double H_r) const {
            return objective(G_l, H_l) + objective(G_r, H_r) - objective(G, H) / 2.0; // TODO: minus gamma ?
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
            return accumulate(gradient.begin(), gradient.end(), 0.0)
                   / (accumulate(hessian.begin(), hessian.end(), 0.0) + _lambda);
        }

        /**
         * Returns an optimal binary split for a given feature index of a Dataset.
         *
         * @param trainSet Training dataset
         * @param gradient Gradient vector
         * @param hessian Hessian vector
         * @param featureId Feature index
         * @return
         */
        SplitInfo optimumGainByFeature(const Dataset &trainSet,
                                  const Vector &gradient,
                                  const Vector &hessian,
                                  int featureId) const {

            double G = accumulate(gradient.begin(), gradient.end(), 0.0);
            double H = accumulate(hessian.begin(), hessian.end(), 0.0);

            double G_l = 0.0, H_l = 0.0, bestGain = std::numeric_limits<double>::min(), bestSplitNumericValue = 0;
            size_t bestSortedIndex = 0;

            // Sort the feature by value and return permutation of indices (i.e., argsort)
            Eigen::RowVectorXi sortedInstanceIds = sortSamplesByFeature(trainSet, featureId);

            // For each feature, compute split gain and keep the split index with maximum gain
            for (size_t i = 0 ; i < trainSet.nRows(); i++){
                G_l += gradient[sortedInstanceIds[i]];
                H_l += hessian[sortedInstanceIds[i]];
                double G_r = G - G_l;
                double H_r = H - H_l;
                double currentGain = calc_split_gain(G, H, G_l, H_l, G_r, H_r);

                if ( currentGain > bestGain) {
                    bestGain = currentGain;
                    bestSplitNumericValue = trainSet.row(sortedInstanceIds[i])(featureId);
                    bestSortedIndex = i + 1;
                }
            }

            // Indices vectors required for split information
            VectorT output(trainSet.nRows());
            for (size_t i = 0 ; i < trainSet.nRows(); i++) {
                output[i] = trainSet.rowIter()[sortedInstanceIds[i]];
            }
            std::vector<size_t> bestLeftInstances(output.data(), output.data() + bestSortedIndex);
            std::vector<size_t> bestLocalLeft(sortedInstanceIds.data(), sortedInstanceIds.data() + bestSortedIndex);
            std::vector<size_t> bestRightInstances(output.data() + bestSortedIndex, output.data() + output.size());
            std::vector<size_t> bestLocalRight(sortedInstanceIds.data() + bestSortedIndex, sortedInstanceIds.data() + sortedInstanceIds.size());

            return SplitInfo(bestGain,
                    bestSplitNumericValue,
                    bestLeftInstances,
                    bestRightInstances,
                    bestLocalLeft,
                    bestLocalRight
            );
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

            // Check if depth is reached
            if (depth > _maxDepth) {
                this->isLeaf = true;
                this->_weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            // Check if # of sample is too small
            if ( trainSet.nRows() <= _minTreeSize) {
                this->isLeaf = true;
                this->_weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            // 1) For each tree node, enumerate over all features:
            // 2) For each feature, sorted the instances by feature numeric value
            //    - Compute gain for every feature (column of design matrix)
            std::vector<SplitInfo> splitInfoPerFeature(trainSet.numFeatures());
            for (size_t featureId = 0; featureId < trainSet.numFeatures(); featureId++) {
                splitInfoPerFeature[featureId] = optimumGainByFeature(trainSet, gradient, hessian, featureId);
            }

            // 3) Use a linear scan to decide the best split along that feature (if categorical perform Mean Target Encoding)
            // 4) Take the best split solution (that maximises gain reduction) over all features
            long bestFeatureId =
                    std::max_element(splitInfoPerFeature.begin(), splitInfoPerFeature.end()) - splitInfoPerFeature.begin();
            SplitInfo bestGain = splitInfoPerFeature[bestFeatureId];

            // Check if best gain is less than minimum split gain (threshold)
            if (bestGain.bestGain() < this->_minSplitGain) {
                this->isLeaf = true;
                this->_weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            this->_splitFeatureIndex = bestFeatureId;
            this->_splitNumericValue = bestGain.splitValue();


            // Recurse on the left subtree
            this->_leftSubTree = std::unique_ptr<TreeNode>(new TreeNode(_lambda, _minSplitGain, _minTreeSize, _maxDepth));
            Dataset leftDataset(trainSet, bestGain, SplitInfo::Side::Left);
            Vector leftGradient = bestGain.split(gradient, SplitInfo::Side::Left);
            Vector leftHessian = bestGain.split(hessian, SplitInfo::Side::Left);
            Vector leftPreviousPreds = bestGain.split(previousPreds, SplitInfo::Side::Left);
            _leftSubTree->build(leftDataset, leftPreviousPreds, leftGradient, leftHessian, shrinkage, depth + 1);

            // Recurse on the right subtree
            this->_rightSubTree = std::unique_ptr<TreeNode>(new TreeNode(_lambda, _minSplitGain, _minTreeSize, _maxDepth));
            Dataset rightDataset(trainSet, bestGain, SplitInfo::Side::Right);
            Vector rightGradient = bestGain.split(gradient, SplitInfo::Side::Right);
            Vector rightHessian = bestGain.split(hessian, SplitInfo::Side::Right);
            Vector rightPreviousPreds = bestGain.split(previousPreds, SplitInfo::Side::Right);
            _rightSubTree->build(rightDataset, rightPreviousPreds, rightGradient, rightHessian, shrinkage, depth + 1);
        }

        /**
         * Return the score for a given sample, i.e. set of features
         *
         * @param sample
         * @return
         */
        double score(const Eigen::RowVectorXd &sample) const {
            if (this->isLeaf) {
                return this->_weight;
            } else if (sample[this->_splitFeatureIndex] < this->_splitNumericValue) {
                return this->_leftSubTree->score(sample);
            } else {
                return this->_rightSubTree->score(sample);
            }
        }
    };
}
