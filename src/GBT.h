#pragma once
#include <vector>
#include<iostream>
#include <memory>
#include <numeric>
#include "dataset.h"
#include "trees/tree.h"
#include "metrics/metric.h"
#include "metrics/logloss.h"
#include "metrics/rmse.h"
#include <chrono>


namespace microgbt {

    /**
     * Gradient Boosting Tree
     */
    class GBT {

    private:

        int _maxDepth, _metricName;
        double _lambda, _gamma, _minSplitGain, _learningRate, _minTreeSize, _shrinkageRate;
        long _bestIteration;
        std::vector<Tree> _trees;
        std::unique_ptr<Metric> _metric;

        /**
         * Return a single decision tree given training data, gradient, hession and shrinkage rate
         *
         * @param dataset
         * @param gradient
         * @param hessian
         * @param shrinkageRate
         */
        Tree buildTree(const Dataset &dataset, const Vector &gradient, const Vector &hessian,
                double shrinkageRate) const {

            Tree tree = Tree(_lambda, _minSplitGain, _minTreeSize, _maxDepth);
            tree.build(dataset, gradient, hessian, shrinkageRate);
            return tree;
        }

    public:

        GBT() = default;

        explicit GBT(const std::map<std::string, double>& params): GBT() {
            this->_lambda = params.at("lambda");
            this->_gamma = params.at("gamma");
            this->_shrinkageRate = params.at("shrinkage_rate");
            this->_minSplitGain = params.at("min_split_gain");
            this->_minTreeSize = params.at("min_tree_size");
            this->_learningRate = params.at("learning_rate");
            this->_maxDepth = static_cast<int>(params.at("max_depth"));
            this->_metricName = static_cast<int>(params.at("metric"));

            if (_metricName == 0){
                this->_metric = std::unique_ptr<Metric>(new LogLoss());
            }
            else {
                this->_metric = std::unique_ptr<Metric>(new RMSE());
            }
        }

        inline int maxDepth() const {
            return _maxDepth;
        }

        inline double lambda() const {
            return _lambda;
        }

        inline double minSplitGain() const {
            return _minSplitGain;
        }

        inline double shrinkageRate() const {
            return _shrinkageRate;
        }

        inline size_t getBestIteration() const {
            return _bestIteration;
        }

        inline double getLearningRate() const {
            return _learningRate;
        }

        /**
         *
         * @param trainX
         * @param trainY
         * @param validX
         * @param validY
         * @param numBoostRound
         * @param earlyStoppingRounds
         */
        void trainPython(const MatrixType& trainX, const Vector& trainY,
                const MatrixType& validX, const Vector& validY,
                int numBoostRound, int earlyStoppingRounds) {
            Dataset trainSet(trainX, trainY);
            Dataset validSet(validX, validY);

            train(trainSet, validSet, numBoostRound, earlyStoppingRounds);

        }

        /**
         * Train a GBT model based on training and validation datasets
         *
         * @param trainSet Input training dataset
         * @param validSet Input validation dataset
         * @param numBoostRound  Number of boosting rounds
         * @param earlyStoppingRounds Early stopping criterion
         */
        void train(const Dataset &trainSet, const Dataset &validSet, int numBoostRound, int earlyStoppingRounds) {

            long bestIteration = 0;
            double learningRate = _shrinkageRate;
            double bestValidationLoss = std::numeric_limits<double>::max();

            // For each iteration, grow an additional tree
            for (long iterCount = 0; iterCount < numBoostRound; iterCount++) {

                std::cout << "[Iteration: " << iterCount << "]" << std::endl;
                auto startTimestamp = std::chrono::high_resolution_clock::now();

                // Current predictions
                Vector predictions = predictDataset(trainSet);

                // Compute gradient and Hessian with respect to prior predictions
                std::cout << "[Computing gradients/Hessians vectors]" << std::endl;
                Vector gradient = _metric->gradients(predictions, trainSet.y());
                Vector hessian = _metric->hessian(predictions);

                // Grow a new tree learner
                std::cout << "[Building next tree...]" << std::endl;
                Tree tree = buildTree(trainSet, gradient, hessian, learningRate);
                std::cout << "[Tree is built successfully]" << std::endl;

                // Update the learning rate
                learningRate *= _learningRate;

                // Append the additional tree
                _trees.push_back(tree);

                // Update train and validation loss
                std::cout << "[Evaluating training / validation losses]" << std::endl;
                Vector trainPreds = predictDataset(trainSet);
                double trainLoss = _metric->lossAt(trainPreds, trainSet.y());
                Vector validPreds = predictDataset(validSet);
                double currentValidationLoss = _metric->lossAt(validPreds, validSet.y());

                auto endTimestamp = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( endTimestamp - startTimestamp ).count();
                std::cout << "[Duration: " << duration << " millis] | [Train Loss]: " << trainLoss
                    << " | [Valid Loss]: " << bestValidationLoss <<std::endl;

                // Update best iteration / best validation error
                if (currentValidationLoss < bestValidationLoss) {
                    bestValidationLoss = currentValidationLoss;
                    bestIteration = iterCount;
                }

                // Check for early stopping
                // Namely, if there is no improvement in the last early_stopping_rounds, then stop
                if (iterCount - bestIteration >= earlyStoppingRounds) {
                    std::cout << "Early stopping, best iteration is:" << bestIteration;
                    std::cout << "Train Loss: " << trainLoss << "| Valid Loss: " << bestValidationLoss <<std::endl;
                    break;
                }
            }

            _bestIteration = bestIteration;
        }

        /**
         * Returns the prediction of a sample
         *
         * @param x Input sample
         * @param numIterations Number of iterations to use for prediction. This is used in case that early stopping took place
         * @return Prediction of input sample
         */
        double predict(const Eigen::RowVectorXd &x, int numIterations) const {
            double score = sumScore(x, numIterations);
            return _metric->scoreToPrediction(score);
        }

        /**
         * Return sum of scores up to numIterations
         *
         * @param x
         * @param trees
         * @param numIterations
         * @return
         */
        double sumScore(const Eigen::RowVectorXd &x, int numIterations) const {
            long double score = 0.0;
            numIterations =  (numIterations == 0) ? _trees.size() : numIterations;
            int limit = 0;
            for (auto &tree: _trees) {
                if (limit < numIterations) {
                    score += tree.score(x);
                    limit++;
                }
                else
                    break;
            }

            return (double)score;
        }

        Vector predictDataset(const Dataset &trainSet) const {
            size_t numSamples = trainSet.nRows();
            size_t numTrees = _trees.size();
            Vector scores(numSamples);
            for (size_t i = 0; i < numSamples; i++) {
                scores[i] = predict(trainSet.row(i), numTrees);
            }

            return scores;
        }
    };
} // namespace microgbt