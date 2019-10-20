#pragma once
#include<vector>
#include<cmath>
#include <Eigen/Dense>

namespace microgbt {

    using Vector = std::vector<double>;

    /**
     * A generic metric that is defined by:
     * * How gradient vector is computed based on current predictions and targets.
     * * How Hessian vector is computed based on current predictions and targets.
     * * Loss evaluation is based on current prediction vector and target vector.
     *
     */
    class Metric {

    public:

        virtual ~Metric() = default;

        /**
         * Compute the gradient vector at given prediction values
         *
         * @param predictions Prediction vector
         * @param targets Vector of values to be predicted
         * @return Gradient vector
         */
        virtual Vector gradients(const Vector &predictions, const Vector &targets) const = 0;

        /**
         * Return the Hessian vecto given prediction vector
         *
         * @param predictions Predictions vector
         * @return Hessian vector
         */
        virtual Vector hessian(const Vector &predictions) const = 0;

        /**
         * Compute the loss at given prediction values.
         *
         * @param predictions Prediction vector
         * @param targets Vector of values to be predicted
         * @return Loss value
         */
        virtual double lossAt(const Vector &predictions, const Vector &targets) const = 0;

        /**
         * Transformation required from Gradient Boosting Trees (GBT) scores to prediction.
         *
         * For example, the logloss will transform the score using the logit function 1 / (1 + exp(-score))
         *
         * @param score Sum of scores over all trees (of GBT)
         * @return Prediction value
         */
        virtual double scoreToPrediction(double score) const = 0;
    };
} // namespace microgbt