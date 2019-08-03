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
#include "numerical_splliter.h"


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

        template<typename T, typename... Args>
        std::unique_ptr<T> make_unique(Args&&... args)
        {
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
        }

    public:

        TreeNode(double lambda, double minSplitGain, double minTreeSize, int maxDepth){
            _lambda = lambda;
            _minSplitGain = minSplitGain;
            _maxDepth = maxDepth;
            _minTreeSize = minTreeSize;
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
                this->weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            // Check if # of sample is too small
            if ( trainSet.nRows() <= _minTreeSize) {
                this->isLeaf = true;
                this->weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            // Initialize a numeric splitter and find best split
            std::unique_ptr<microgbt::Splitter> splitter = make_unique<microgbt::NumericalSplitter>(_lambda);
            SplitInfo bestGain = splitter->findBestSplit(trainSet, gradient, hessian);

            // Check if best gain is less than minimum split gain (threshold)
            if (bestGain.bestGain() < this->_minSplitGain) {
                this->isLeaf = true;
                this->weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            this->splitFeatureIndex = bestGain.getBestFeatureId();
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
