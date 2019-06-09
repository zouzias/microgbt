#pragma once
#include<vector>
#include<cmath>
#include <Eigen/Dense>

namespace microgbt {

    using Vector = std::vector<double>;

    /**
     * A generic metric that is defined by:
     * * How gradient is computed based on current prediction vector and target vector.
     * * How Hessian is computed based on current prediction vector and target vector.
     * * Loss evaluation based on current prediction vector and target vector.
     *
     */
    class Metric {

    public:

        virtual ~Metric() = default;

        /**
         * Compute the gradient at given predictions vector
         *
         * @param predictions
         * @param labels
         * @return
         */
        virtual Vector gradients(const Vector &scores, const Vector &labels) const = 0;

        /**
         * Return the Hessian vector at given predictions vector
         *
         * @param predictions
         * @param labels
         * @return
         */
        virtual Vector hessian(const Vector &scores) const = 0;

        /**
         * Compute the loss at given predictions.
         *
         * @param predictions
         * @param labels
         * @return
         */
        virtual double lossAt(const Vector &scores, const Vector &labels) const = 0;

        /**
         * Transformation required from Gradient Boosting Trees scores to final prediction
         *
         * @param score Sum of scores over all trees (of GBT)
         * @return
         */
        virtual double scoreToPrediction(double score) const = 0;
    };
} // namespace microgbt