#pragma once

#include <vector>
#include <string>
#include <map>
#include <queue>
#include <memory>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "treenode.h"
#include "../metrics/metric.h"
#include "class_list.h"
#include "tree_builder_state.h"


namespace microgbt {

    using FeatureId = long;
    using NodeId = long;
    using GradientHessianPair = std::pair<double, double>;

    /**
     * A binary tree that contains information regarding Gradient Boosting a la xgboost (gradients / hessians information)
     */
    class Tree {

        /** Maximum depth of tree */
        int _maxDepth;

        long _minTreeSize;

        /** Gradient boosting parameters */
        double _lambda, _minSplitGain;

        /** Root of tree */
        std::shared_ptr<TreeNode> root;

        std::vector<std::shared_ptr<TreeNode>> nodes;
    public:

        Tree(double lambda, double minSplitGain, size_t minTreeSize, int maxDepth):
        nodes(0) {
            _lambda = lambda;
            _minSplitGain = minSplitGain;
            _maxDepth = maxDepth;
            _minTreeSize = minTreeSize;
        }

        std::shared_ptr<TreeNode> newTreeNode(size_t numSamples, size_t size) {
            long nodeId = nodes.size();
            TreeNode node(nodeId, _lambda, numSamples, size);
            node.makeLeaf();
            nodes.push_back(std::make_shared<TreeNode>(node));
            return nodes[nodeId];
        }

         /**
          * Build a tree using 'optimal' binary splits
          * based on gradient & Hessian vectors (following xgboost's formulation).
          *
          * The tree building approach is described in SLIQ: A fast scalable classifier for data mining
          * (https://doi.org/10.1007/BFb0014141)
          *
          * @param dataset Training dataset
          * @param gradient Vector of gradients to fit on
          * @param hessian Vector of second derivatives, Hessians to fit on
          * @param shrinkage Shrinkage additive boosting rate
          */
        void build(const Dataset &dataset,
                   const Vector &gradient,
                   const Vector &hessian,
                   double shrinkage) {

            long numSamples = dataset.nRows();
            // Create the root node
            root = newTreeNode(numSamples, numSamples);

            // SLIQ classlist
            ClassList classList(numSamples);

            // Sum the gradients / hessians over all samples
            double G = std::accumulate(gradient.begin(), gradient.end(), 0.0);
            double H = std::accumulate(hessian.begin(), hessian.end(), 0.0);

            // Root node setup: gradient/hessian/weight
            root->setLambda(_lambda);
            root->setGradientSum(G);
            root->setHessianSum(H);
            root->setWeight(root->calc_leaf_weight());

            // Build tree breadth first search (BFS) as in SLIQ
            // That is, go over each depth level of the tree
            for (int depth = 0; depth < _maxDepth ; depth ++) {
                std::cout << "[Working on depth '" << depth << "']" << std::endl;

                for (auto& node : nodes){
                    if (node->isLeaf()) {
                        classList.zero(node->getNodeId());
                    }
                }

                // Keep track of partial sums (Gradients / Hessians) per leaf node
                TreeBuilderState state(nodes.size());

                // Go over all features to compute optimal gain
                for (FeatureId featureIdx = 0; featureIdx < dataset.numFeatures(); featureIdx++) {
                    std::cout << "[Working on feature '" << featureIdx << " out of " << dataset.numFeatures() << "']" << std::endl;

                    // Instantiate the tree builder state (SLIQ)
                    state.zeroAllPartialSums();

                    // Clean the list of candidate left indices per leaf node
                    classList.zero();

                    const Permutation& perm = dataset.sortedColumnIndices(featureIdx);

                    // Go over all pre-sorted sample indices: 'sampleIdx'
                    for (NodeId i = 0; i < numSamples; i++) {
                        size_t sampleIdx = perm(i);
                        double sortedFeatureValue = dataset(sampleIdx, featureIdx);
                        double g = gradient[sampleIdx];
                        double h = hessian[sampleIdx];
                        NodeId leafId = classList.nodeAt(sampleIdx);

                        // Calculate gain on possible split on leafId
                        // If gain is better than before on leaf “leafId”, mark it

                        // Add gradient and Hessian to partial sums
                        state.addToPartialSums(leafId, g, h);
                        GradientHessianPair partialSums = state.partialSums(leafId);
                        double gain = nodes[leafId]->calc_split_gain(partialSums.first, partialSums.second);

                        // Assign sample with index sampleIdx to class list
                        classList.increaseLeftSizeByNode(leafId);

                        // If gain is better, mark it
                        if (gain > nodes[leafId]->bestGain()
                                 && gain > 0
                                 && classList.getLeftSize(leafId) > _minTreeSize
                                 && classList.getRightSize(leafId) > _minTreeSize
                                ) {
                            nodes[leafId]->updateWeight(shrinkage);
                            nodes[leafId]->setLeftSize(classList.getLeftSize(leafId));
                            nodes[leafId]->setBestSplitValue(sortedFeatureValue);
                            nodes[leafId]->setBestGain(gain);
                            nodes[leafId]->setBestSplitFeatureId(featureIdx);
                            nodes[leafId]->setLeftGradientSum(partialSums.first);
                            nodes[leafId]->setLeftHessianSum(partialSums.second);
                        }
                    }
                }

                // For each node, go and reconstruct the left sample indices
                for (auto& node: nodes) {
                    long featureIdx = node->getBestSplitFeatureId();
                    long leftSize = node->getLeftSize();
                    if (leftSize == 0) {
                        continue;
                    }

                    NodeId nodeId = node->getNodeId();
                    std::vector<long> leftIndices(leftSize);
                    const Permutation& perm = dataset.sortedColumnIndices(featureIdx);

                    // Go over all pre-sorted sample indices: 'sampleIdx'
                    long idx = 0;
                    while( leftSize > 0 ){
                        size_t sampleIdx = perm(idx);
                        NodeId classListNodeId = classList.nodeAt(sampleIdx);
                        if (classListNodeId == nodeId) {
                            node->setLeftSampleId(sampleIdx);
                        }

                        idx++;
                        leftSize--;
                    }
                }

                // At this point, leaves nodes contain all information to decide
                // if a split will be performed or not

                // Go over all leaves, say “l”
                // Create the left and right sub-trees based on leaf info
                size_t n = nodes.size();
                for (size_t i = 0; i < n; i++) {

                    if (nodes[i]->getSize() < _minTreeSize){
                        nodes[i]->updateWeight(shrinkage);
                        continue;
                    }

                    if (nodes[i]->getLeftSize() < _minTreeSize){
                        nodes[i]->updateWeight(shrinkage);
                        continue;
                    }

                    if (nodes[i]->getRightSize() < _minTreeSize) {
                        nodes[i]->updateWeight(shrinkage);
                        continue;
                    }

                    if (nodes[i]->bestGain() < _minSplitGain){
                        nodes[i]->updateWeight(shrinkage);
                        continue;
                    }

                    if (nodes[i]->isLeaf()) {
                        nodes[i]->setLeftSubTree(newTreeNode(numSamples, nodes[i]->getLeftSize()), shrinkage);
                        nodes[i]->setRightSubTree(newTreeNode(numSamples, nodes[i]->getRightSize()), shrinkage);
                        nodes[i]->markInnerNode();
                    }
                }

                // Go over leaves, say “l”
                // Update each “previous leaf” with the left or right leaf pointer
                for (long i = 0; i < numSamples; i++) {
                    NodeId leafId = classList.nodeAt(i);
                    if (!nodes[leafId]->isLeaf()) {
                        if (nodes[leafId]->isLeftAssigned(i)) {
                            classList.updateNodeId(i, nodes[leafId]->getLeftSubTreeId());
                        } else {
                            classList.updateNodeId(i, nodes[leafId]->getRightSubTreeId());
                        }
                    }
                }
            }
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
