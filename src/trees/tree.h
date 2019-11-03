#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "treenode.h"
#include "split_info.h"
#include "../metrics/metric.h"


namespace microgbt {

    /**
     * A Decision/Regression Tree with binary splits
     */
    class Tree {
    private:
        /** Maximum depth of tree */
        int _maxDepth;

        /** Gradient boosting parameters */
        double _lambda, _minSplitGain, _minTreeSize;

        /** Root of tree */
        std::shared_ptr<TreeNode> root;

    public:

        Tree(double lambda, double minSplitGain, double minTreeSize, int maxDepth) {
            _lambda = lambda;
            _minSplitGain = minSplitGain;
            _maxDepth = maxDepth;
            _minTreeSize = minTreeSize;
        }

         /**
          * Recursively (and greedily) build decision tree using 'optimal' binary splits
          * based on gradient & Hessian vectors.
          *
          * @param train_set Training dataset
          * @param previousPreds Previous iteration predictions
          * @param gradient Vector of gradients
          * @param hessian Vector of second derivatives, Hessian
          * @param shrinkage Shrinkage rate
          */
        void build(const Dataset &trainSet, const Vector &previousPreds,
                   const Vector &gradient,
                   const Vector &hessian,
                   double shrinkage) {

            this->root = std::unique_ptr<TreeNode>(new TreeNode(_lambda, _minSplitGain, _minTreeSize, _maxDepth));
            int depth = 0;
            this->root->build(trainSet, previousPreds, gradient, hessian, shrinkage, depth);
        }

        /**
         * Return tree score for a sample point, i.e., set of features
         *
         * @param sample A sample as row vector
         * @return Score of tree
         */
        double score(const Eigen::RowVectorXd &sample) const {
            return root->score(sample);
        }
    };
} // namespace microgbt