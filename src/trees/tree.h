#pragma once

#include <vector>

#include "treenode.h"
#include "split_info.h"
#include "../metrics/metric.h"

namespace microgbt
{

/**
     * A decision / regression tree with binary splits
     */
class Tree
{

    // Maximum depth of tree
    int _maxDepth;

    // Gradient boosting parameters
    double _lambda, _minSplitGain, _minTreeSize;

    // Root of tree
    std::shared_ptr<TreeNode> _root;

public:
    Tree(double lambda, double minSplitGain, double minTreeSize, int maxDepth)
    {
        _lambda = lambda;
        _minSplitGain = minSplitGain;
        _maxDepth = maxDepth;
        _minTreeSize = minTreeSize;
    }

    /**
          * Recursively (and greedily) build regression tree using 'optimal greedy' binary splits
          * based on gradient & Hessian vectors.
          *
          * @param trainSet Training dataset
          * @param previousPreds Prediction based on previous trees
          * @param gradient Gradient vector
          * @param hessian Vector of second derivatives, Hessian
          * @param shrinkage Shrinkage rate
          */
    void build(const Dataset &trainSet,
               const std::vector<Histogram>& histograms,
               const Vector &previousPreds,
               const Vector &gradient,
               const Vector &hessian,
               double shrinkage) {

        this->_root = std::make_shared<TreeNode>(_lambda, _minSplitGain, _minTreeSize, _maxDepth);
        int depth = 0;
        this->_root->build(trainSet, histograms, previousPreds, gradient, hessian, shrinkage, depth);
    }

    /**
         * Return tree score for a sample point, i.e., set of features
         *
         * @param sample A sample as row vector
         * @return Score of tree
         */
    double score(const Eigen::RowVectorXd &sample) const { return _root->score(sample); }
};
} // namespace microgbt