#pragma once
#include <vector>
#include<iostream>
#include <memory>
#include <omp.h>
#include "dataset.h"
#include "trees/tree.h"
#include "metrics/metric.h"
#include "metrics/logloss.h"
#include "metrics/rmse.h"


namespace microgbt {

    using Vector = std::vector<double>;

    /**
     * A Gradient Boosting Tree
     */
    class GBT {

    private:

        int _maxDepth, _metricName;
        double _lambda, _gamma, _minSplitGain, _learningRate, _minTreeSize, _shrinkageRate;
        size_t _bestIteration;
        std::vector<Tree> _trees;
        std::unique_ptr<Metric> _metric;

        /**
         * Return a single decision tree given training data, gradient, hession and shrinkage rate
         *
         * @param trainSet
         * @param gradient
         * @param hessian
         * @param shrinkageRate
         */
        Tree buildTree(Dataset &trainSet, const Vector& previousPreds, const Vector &gradient,
                const Vector &hessian,
                       double shrinkageRate) const {


            Tree tree = Tree(_lambda, _minSplitGain, _minTreeSize, _maxDepth);
            tree.build(trainSet, previousPreds, gradient, hessian, shrinkageRate);
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
        void trainPython(const Eigen::MatrixXd& trainX, Vector& trainY,
                const Eigen::MatrixXd& validX, Vector& validY,
                int numBoostRound, int earlyStoppingRounds) {
            Dataset trainSet(trainX, trainY);
            Dataset validSet(validX, validY);

            train(trainSet, validSet, numBoostRound, earlyStoppingRounds);

        }


        void train(Dataset &trainSet, Dataset &validSet, int numBoostRound, int earlyStoppingRounds) {

            // Allow nested threading in OpenMP
            omp_set_nested(1);

            std::vector<Tree> trees;
            long bestIteration = 0;
            double learningRate = _shrinkageRate;
            double bestValidationLoss = std::numeric_limits<double>::max();

            // For each iteration, grow an additional tree
            for (long iterCount = 0; iterCount < numBoostRound; iterCount++) {

                std::cout << "Iteration: " << iterCount << std::endl;

                // Current predictions
                Vector scores = predictDatasetFromTrees(trainSet, trees);

                // Compute gradient and Hessian with respect to prior predictions
                Vector gradient = _metric->gradients(scores, trainSet.y());
                Vector hessian = _metric->hessian(scores);

                // Grow a new tree learner
                Tree tree = buildTree(trainSet, scores, gradient, hessian, learningRate);

                // Update the learning rate
                learningRate *= _learningRate;

                // Append the additional tree
                trees.push_back(tree);

                // Update train and validation loss
                Vector trainPreds = predictDatasetFromTrees(trainSet, trees);
                double trainLoss = _metric->lossAt(trainPreds, trainSet.y());
                Vector validPreds = predictDatasetFromTrees(validSet, trees);
                double currentValidationLoss = _metric->lossAt(validPreds, validSet.y());

                std::cout << "[Train Loss]: " << trainLoss << " | [Valid Loss]: " << bestValidationLoss <<std::endl;

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

            _trees = trees;
            _bestIteration = bestIteration;
        }

        /**
         * Returns the prediction probability of a sample
         *
         * @param x A sample
         * @param trees Vector of trees
         * @param numIterations Number of iterations to use for prediction. This is used in case of early stopping
         * @return
         */
        double predict(const Eigen::RowVectorXd &x, int numIterations) const {
            double score = sumScore(x, numIterations);
            return _metric->scoreToPrediction(score);
        };

        /**
         * Return sum of scores up to numIterations
         *
         * @param x
         * @param trees
         * @param numIterations
         * @return
         */
        double sumScore(const Eigen::RowVectorXd &x, int numIterations) const {
            double score = 0.0;
            int limit = 0;
            for (auto &tree: _trees) {
                if (limit < numIterations) {
                    score += tree.score(x);
                    limit++;
                }
                else
                    break;
            }
            return score;
        }

        double predictFromTrees(const Eigen::RowVectorXd &x, const std::vector<Tree> &trees) const {
            size_t n = trees.size();
            double score = 0.0;
            #pragma omp parallel for default(none) shared(n, x, trees) reduction(+: score)
            for (size_t i = 0; i < n; i ++){
                score += trees[i].score(x);
            }
            return _metric->scoreToPrediction(score);
        }

        Vector predictDatasetFromTrees(const Dataset &trainSet, const std::vector<Tree> &trees) const {
            size_t numSamples = trainSet.nRows();
            Vector scores(numSamples);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < numSamples; i++) {
                scores[i] = predictFromTrees(trainSet.row(i), trees);
            }

            return scores;
        }
    };
} // namespace microgbt