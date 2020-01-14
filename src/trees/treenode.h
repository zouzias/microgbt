#pragma once
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <iostream>
#include <thread>

#include "../dataset.h"
#include "split_info.h"
#include "numerical_splliter.h"
#include "../types.h"

namespace microgbt
{

/**
     * A node of a regression tree of GBT
     */
    class TreeNode {

        // Maximum tree depth / minimum tree size
        int _maxDepth, _minTreeSize;

        // Regularization parameters
        double _lambda, _minSplitGain;

        // Is the node a leaf?
        bool _isLeaf;

        // Pointers to left and right subtrees
        std::unique_ptr<TreeNode> leftSubTree, rightSubTree;

        // Feature index on which the split took place
        long _splitFeatureIndex;

        // Numeric value on which the binary tree split took place
        double _splitNumericValue, _weight;

    public:

        explicit TreeNode(double lambda, double minSplitGain, int minTreeSize, int maxDepth){
            _lambda = lambda;
            _minSplitGain = minSplitGain;
            _maxDepth = maxDepth;
            _minTreeSize = minTreeSize;
            _splitFeatureIndex = -1;
            _isLeaf = false;
            _splitNumericValue = std::numeric_limits<double>::min();
            _weight = 0.0;
        }

         /**
          * Return  the optimal weight of a leaf node.
          * (Refer to Eq5 of Reference[1])
          *
          * @param gradient Gradient vector
          * @param hessian Hessian vector
          * @return
          */
    inline double calc_leaf_weight(const Vector &gradient,
                                   const Vector &hessian) const
    {
        return -std::accumulate(gradient.begin(), gradient.end(), 0.0) / (std::accumulate(hessian.begin(), hessian.end(), 0.0) + _lambda);
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
         * @param trainSet Train dataset
         * @param gradient Gradient vector
         * @param hessian Hessian vector
         * @param shrinkage Current shrinkage parameter
         * @param depth Current depth on building process
         */
        void build(const Dataset &trainSet,
                   const VectorD &gradient,
                   const VectorD &hessian,
                   double shrinkage,
                   int depth) {

            // Check if depth is reached
            if (depth > _maxDepth) {
                this->_isLeaf = true;
                this->_weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            // Check if # of sample is too small
            if ( trainSet.nRows() <= _minTreeSize) {
                this->_isLeaf = true;
                this->_weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            // Initialize a numeric splitter and find best split
            NumericalSplitter splitter(_lambda);
            SplitInfo bestGain = splitter.findBestSplit(trainSet);

            // Check if best gain is less than minimum split gain (threshold)
            if (bestGain.bestGain() < this->_minSplitGain) {
                this->_isLeaf = true;
                this->_weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            // Update feature index and numeric value of optimal greedy split
            this->_splitFeatureIndex = bestGain.getBestFeatureId();
            this->_splitNumericValue = bestGain.splitValue();

            // Recurse on left and right subtree
            Dataset leftDataset(trainSet, bestGain, SplitInfo::Side::Left);
            VectorD leftGradient = bestGain.split(gradient, SplitInfo::Side::Left);
            VectorD leftHessian = bestGain.split(hessian, SplitInfo::Side::Left);

            // Create Histogram on left subtree
            for (long j = 0 ; j < leftDataset.numFeatures(); j++){
                leftDataset.histogram(j)->fillValues(leftDataset.col(j), leftGradient, leftHessian);
            }

            this->leftSubTree = std::make_unique<TreeNode>(_lambda, _minSplitGain, _minTreeSize, _maxDepth);
            leftSubTree->build(leftDataset, leftGradient, leftHessian, shrinkage, depth + 1);

            Dataset rightDataset(trainSet, bestGain, SplitInfo::Side::Right);
            VectorD rightGradient = bestGain.split(gradient, SplitInfo::Side::Right);
            VectorD rightHessian = bestGain.split(hessian, SplitInfo::Side::Right);

            // Create Histogram on right subtree
            for (long j = 0 ; j < rightDataset.numFeatures(); j++){
                rightDataset.histogram(j)->fillValues(rightDataset.col(j), rightGradient, rightHessian);
            }

            this->rightSubTree = std::make_unique<TreeNode>(_lambda, _minSplitGain, _minTreeSize, _maxDepth);
            rightSubTree->build(rightDataset, rightGradient, rightHessian, shrinkage, depth + 1);
        }

    /**
         * Return the score for a given sample, i.e. set of features
         *
         * @param sample
         * @return
         */
    double score(const Eigen::RowVectorXd &sample) const
    {
        if (this->_isLeaf)
        {
            return this->_weight;
        }
        else if (sample[this->_splitFeatureIndex] < this->_splitNumericValue)
        {
            return this->leftSubTree->score(sample);
        }
        else
        {
            return this->rightSubTree->score(sample);
        }
    }
};
} // namespace microgbt
