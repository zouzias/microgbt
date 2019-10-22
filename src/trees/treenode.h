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
#include "numerical_splliter.h"
#include "../types.h"


namespace microgbt {

    /**
     * A node of a regression tree of GBT
     */
    class TreeNode {

        // Maximum tree depth
        int _maxDepth;

        // Regularization parameters
        double _lambda, _minSplitGain, _minTreeSize;

        // Is the node a leaf?
        bool _isLeaf;

        // Pointers to left and right subtrees
        std::unique_ptr<TreeNode> leftSubTree, rightSubTree;

        // Feature index on which the split took place
        long _splitFeatureIndex;

        // Numeric value on which the binary tree split took place
        double _splitNumericValue, _weight;

        template<typename T, typename... Args>
        std::unique_ptr<T> make_unique(Args&&... args)
        {
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
        }

    public:

        explicit TreeNode(double lambda, double minSplitGain, double minTreeSize, int maxDepth){
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
                                       const Vector &hessian) const {
            return - std::accumulate(gradient.begin(), gradient.end(), 0.0)
                   / (std::accumulate(hessian.begin(), hessian.end(), 0.0) + _lambda);
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
         * @param previousPreds
         * @param gradient Gradient vector
         * @param hessian Hessian vector
         * @param shrinkage Current shrinkage parameter
         * @param depth Current depth on building process
         */
        void build(const Dataset &trainSet,
                   const Vector &previousPreds,
                   const Vector &gradient,
                   const Vector &hessian,
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
            std::unique_ptr<microgbt::Splitter> splitter = make_unique<microgbt::NumericalSplitter>(_lambda);
            SplitInfo bestGain = splitter->findBestSplit(trainSet, gradient, hessian);

            // Check if best gain is less than minimum split gain (threshold)
            if (bestGain.bestGain() < this->_minSplitGain) {
                this->_isLeaf = true;
                this->_weight = this->calc_leaf_weight(gradient, hessian) * shrinkage;
                return;
            }

            // Update feature index and numeric value of optimal greedy split
            this->_splitFeatureIndex = bestGain.getBestFeatureId();
            this->_splitNumericValue = bestGain.splitValue();

            #pragma omp parallel sections default(none) shared(bestGain, shrinkage, depth)
            {
                // Recurse on the left subtree
                #pragma omp section
                {
                    Dataset leftDataset(trainSet, bestGain, SplitInfo::Side::Left);
                    Vector leftGradient = bestGain.split(gradient, SplitInfo::Side::Left);
                    Vector leftHessian = bestGain.split(hessian, SplitInfo::Side::Left);
                    Vector leftPreviousPreds = bestGain.split(previousPreds, SplitInfo::Side::Left);
                    this->leftSubTree = std::unique_ptr<TreeNode>(
                            new TreeNode(_lambda, _minSplitGain, _minTreeSize, _maxDepth));
                    leftSubTree->build(leftDataset, leftPreviousPreds, leftGradient, leftHessian, shrinkage, depth + 1);
                }


                // Recurse on the right subtree
                #pragma omp section
                {
                    Dataset rightDataset(trainSet, bestGain, SplitInfo::Side::Right);
                    Vector rightGradient = bestGain.split(gradient, SplitInfo::Side::Right);
                    Vector rightHessian = bestGain.split(hessian, SplitInfo::Side::Right);
                    Vector rightPreviousPreds = bestGain.split(previousPreds, SplitInfo::Side::Right);

                    this->rightSubTree = std::unique_ptr<TreeNode>(
                            new TreeNode(_lambda, _minSplitGain, _minTreeSize, _maxDepth));
                    rightSubTree->build(rightDataset, rightPreviousPreds, rightGradient, rightHessian, shrinkage, depth + 1);
                }
            }
        }

        /**
         * Return the score for a given sample, i.e. set of features
         *
         * @param sample
         * @return
         */
        double score(const Eigen::RowVectorXd &sample) const {
            if (this->_isLeaf) {
                return this->_weight;
            } else if (sample[this->_splitFeatureIndex] < this->_splitNumericValue) {
                return this->leftSubTree->score(sample);
            } else {
                return this->rightSubTree->score(sample);
            }
        }
    };
}
