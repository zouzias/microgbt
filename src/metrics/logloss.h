#pragma once

#include<algorithm>
#include<random>
#include "metric.h"

namespace microgbt {

    /**
     * Log loss metric (a.k.a. logistic loss)
     *
     * Logistic loss: y_i ln(1 + exp(-pred_i)) + (1-y_i) ln( 1 + exp(pred_i))
     *
     * Reference: https://en.wikipedia.org/wiki/Loss_functions_for_classification#Logistic_loss
     */
    class LogLoss :
            public Metric {

    private:

        /*
         * Numerical tolerance on boundary of log(x) and log(1-x) function in range [0,1]
         */
        double _eps;

    public:

        LogLoss() {
            _eps = 10e-8;
        }

        /**
         * Clips value in numeric interval [_eps, 1 - _eps]
         *
         * @param value
         * @return
         */
        inline double clip(double value) const {
            if ( value > 1 - _eps )
                return 1 - _eps;

            if ( value < _eps)
                return _eps;

            return value;
        }

        static inline double logit(double score) {
            return 1.0 / (1 + exp(-score));
        }

        Vector gradients(const Vector &predictions, const Vector &labels) const override {
            Vector gradients;
            std::transform(labels.begin(), labels.end(),
                           predictions.begin(), std::back_inserter(gradients), std::minus<double>());

            return gradients;
        }

        Vector hessian(const Vector &predictions) const override {
            Vector hessians;

            std::transform(predictions.begin(), predictions.end(),
                           std::back_inserter(hessians),
                           []( double prediction) {
                return abs(logit(prediction)) * ( 1- abs(prediction));
            });

            return hessians;
        }

        double lossAt(const Vector &scores, const Vector &y) const override {
            double loss = 0.0;

            size_t n = scores.size();
            for (size_t i = 0; i< n; i ++){
                loss += y[i] * log(clip(scores[i])) + (1 - y[i]) * log(1 - clip(scores[i]));
            }

            return -loss / n;
        }

        double scoreToPrediction(double score) const override {
            return logit(score);
        }

    };
}