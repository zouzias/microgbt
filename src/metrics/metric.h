#pragma once
#include<vector>
#include<cmath>
#include <Eigen/Dense>

namespace microgbt {

    using Vector = std::vector<double>;

    /**
     * A generic metric is defined by:
     * 1) gradient method: How gradient is computed based on current prediction vector and target vector.
     * 2) hessian method: How Hessian is computed based on current prediction vector.
     * * Loss evaluation based on current prediction vector and target vector.
     *
     */
    class Metric {

    public:

        virtual ~Metric() = default;

        /**
         * Return the gradient of the metric at given predictions vector
         *
         * @param predictions
         * @param labels
         * @return
         */
        virtual Vector gradients(const Vector &scores, const Vector &labels) const = 0;

        /**
         * Return the Hessian vector of the metric at given predictions vector
         *
         * @param predictions
         * @param labels
         * @return
         */
        virtual Vector hessian(const Vector &predictions) const = 0;

        /**
         * Evaluates the loss function at given prediction and target vector.
         *
         * @param predictions
         * @param labels
         * @return
         */
        virtual double lossAt(const Vector &scores, const Vector &labels) const = 0;

        /**
         * Transformation used to convert Gradient Boosting Trees scores to prediction (classification or regression)
         *
         * @param score Sum of scores over all Gradient Boosting Trees
         * @return GBT prediction
         */
        virtual double scoreToPrediction(double score) const = 0;
    };
}