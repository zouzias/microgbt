#pragma once
#include<memory>
#include<map>
#include <utility>
#include<vector>
#include<string>
#include<algorithm>
#include<numeric>
#include<iterator>
#include<iostream>
#include <thread>
#include <set>

#include "../dataset.h"
#include "class_list.h"


namespace microgbt {

    /**
     * A node of a binary tree
     */
    class TreeNode {

        NodeId _nodeId;
        double _lambda;
        std::shared_ptr<TreeNode> leftSubTree;
        std::shared_ptr<TreeNode> rightSubTree;

        // Feature id on which the best split happened on the current node
        long _bestSplitFeatureId;

        // Numeric value on which a binary tree split took place
        double _bestSplitNumericValue, _weight = 0;

        // Total gradient and Hessian sum at current node
        double _gradientSum, _hessianSum;

        double _bestGain, _leftGradientSum, _leftHessianSum;

        // Is TreeNode a leaf?
        bool _isLeaf = true;

        // Number of samples assigned to current node
        long _size;

        // Size of left sub-tree
        long _leftSize;

        // Set of sample indices that corresponds to left subtree
        std::vector<bool> _leftSampleIds;

    public:

        TreeNode(long nodeId, double lambda, size_t numSamples, size_t nodeSize):
                _leftSampleIds(numSamples, false){
            _nodeId = nodeId;
            _lambda = lambda;
            _bestGain = std::numeric_limits<double>::lowest();
            _bestSplitNumericValue = std::numeric_limits<double>::lowest();
            _bestSplitFeatureId = -1;
            _gradientSum = 0.0, _hessianSum = 0.0;
            _leftGradientSum = 0.0, _leftHessianSum = 0.0;
            _weight = 0.0;
            _leftSize = 0;
            _size = nodeSize;
            _isLeaf = true;
            leftSubTree = nullptr, rightSubTree = nullptr;
        }

        void markInnerNode() {
            _isLeaf = false;
        }

        void makeLeaf() {
            _isLeaf = true;
        }

        inline bool isLeaf() const{
            return _isLeaf;
        }

        inline double bestGain() const {
            return _bestGain;
        }

        void setBestGain(double gain) {
            _bestGain = gain;
        }

        void setLeftSize(long size){
            _leftSize = size;
        }

        void setLeftSampleId(long leftSampleId){
            _leftSampleIds[leftSampleId] = true;
        }

        bool isLeftAssigned(size_t sampleId) {
            return _leftSampleIds[sampleId];
        }

        void setLeftGradientSum(double value) {
            _leftGradientSum = value;
        }

        void updateWeight(double shrinkage) {
            _weight = calc_leaf_weight() * shrinkage;
        }

        void setLeftHessianSum(double value) {
            _leftHessianSum = value;
        }

        void setBestSplitFeatureId(long id) {
            _bestSplitFeatureId = id;
        }

        long getBestSplitFeatureId() const {
            return _bestSplitFeatureId;
        }

        void setWeight(double weight) {
            _weight = weight;
        }

        long getSize() const {
            return _size;
        }

        void setBestSplitValue(double bestSplitValue) {
            _bestSplitNumericValue = bestSplitValue;
        }

        void setLambda(double lambda) {
            _lambda = lambda;
        }

        void setGradientSum(double gradientSum) {
            _gradientSum = gradientSum;
        }

        void setHessianSum(double hessianSum) {
            _hessianSum = hessianSum;
        }

        inline double getRightGradientSum() const {
            return _gradientSum - _leftGradientSum;
        }

        inline double getRightHessianSum() const {
            return std::max(_hessianSum - _leftHessianSum, 0.0);
        }

        inline double getGradientSum() const {
            return _gradientSum;
        }

        inline double getHessianSum() const {
            return _hessianSum;
        }

        inline NodeId getLeftSubTreeId() const {
            if (leftSubTree == nullptr) {
                return -1;
            }
            return leftSubTree->getNodeId();
        }

        inline NodeId getRightSubTreeId() const {
            if (rightSubTree == nullptr) {
                return -1;
            }
            return rightSubTree->getNodeId();
        }

        inline NodeId getNodeId() const {
            return _nodeId;
        }

        inline long getLeftSize() const {
            return _leftSize;
        }

        inline long getRightSize() const {
            return _size - getLeftSize();
        }

        void zeroLeftBitset() {
            std::fill(_leftSampleIds.begin(), _leftSampleIds.end(), false);
        }

        void setLeftSubTree(const std::shared_ptr<TreeNode>& treeNodePtr, double shrinkage) {
            leftSubTree = treeNodePtr;
            leftSubTree->setLambda(_lambda);
            leftSubTree->setGradientSum(_leftGradientSum);
            leftSubTree->setHessianSum(_leftHessianSum);
            leftSubTree->zeroLeftBitset();
            leftSubTree->_leftGradientSum = 0.0;
            leftSubTree->_leftHessianSum = 0.0;
            leftSubTree->makeLeaf();
            leftSubTree->updateWeight(shrinkage);
        }

        void setRightSubTree(const std::shared_ptr<TreeNode>& treeNodePtr, double shrinkage) {
            rightSubTree = treeNodePtr;
            rightSubTree->setLambda(_lambda);
            rightSubTree->setGradientSum(getRightGradientSum());
            rightSubTree->setHessianSum(getRightHessianSum());
            rightSubTree->zeroLeftBitset();
            rightSubTree->_leftGradientSum = 0.0;
            rightSubTree->_leftHessianSum = 0.0;
            rightSubTree->makeLeaf();
            rightSubTree->updateWeight(shrinkage);
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
        inline double calc_leaf_weight() const {
            return - _gradientSum / (_hessianSum + _lambda);
        }

        /**
         * Returns objective value for a given gradient, Hessian and lambda value (gradient boosting parameters)
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
        constexpr double calc_split_gain(double G_l, double H_l) const {
            return objective(G_l, H_l) + objective(_gradientSum - G_l, _hessianSum - H_l)
            - objective(_gradientSum, _hessianSum) / 2.0; // TODO: minus \gamma
        }


        /**
         * Return the score for a given sample, i.e. set of features
         *
         * @param sample
         * @return
         */
        double score(const Eigen::RowVectorXd &sample) const {
            if (_isLeaf) {
                return _weight;
            } else if (sample[_bestSplitFeatureId] < _bestSplitNumericValue) {
                return leftSubTree->score(sample);
            } else {
                return rightSubTree->score(sample);
            }
        }

        /**
         * Returns a list of string containing all subtree information in DOT (a graph description language)
         *
         * See: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
         * @return
         */
        std::vector<std::string> toDigraph() const {
            std::vector<std::string> output;
            if (isLeaf()) {
                return output;
            }

            NodeId id = this->getNodeId();
            NodeId left = this->getLeftSubTreeId();
            output.push_back(std::to_string(id) + " [label=\"" + print() + "\"]");
            output.push_back(std::to_string(id) + " -> " + std::to_string(left));

            NodeId right = this->getRightSubTreeId();
            output.push_back(std::to_string(id) + " -> " + std::to_string(right));


            auto leftCollection = leftSubTree->toDigraph();
            auto rightCollection = rightSubTree->toDigraph();
            std::copy(begin(leftCollection), end(leftCollection), std::back_inserter(output));
            std::copy(begin(rightCollection), end(rightCollection), std::back_inserter(output));

            return output;
        }

        std::string print() const {
            std::stringstream ss;
            ss << " | Node id: " << getNodeId();
            ss << " | Best Gain: " << bestGain();
            ss << " | Weight: " << calc_leaf_weight();
            ss << " | Left size: " << getLeftSize();
            ss  << " | S: " << getSize();
            ss << " | G sum: " << getGradientSum();
            ss << " | H sum: " << getHessianSum();
            return ss.str();
        }
    };
} // namespace microgbt
