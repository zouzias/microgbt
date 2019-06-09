#pragma once

#include<algorithm>
#include<random>
#include<cmath>
#include "metric.h"

namespace microgbt {

    using Vector = std::vector<double>;

    /**
     * Log loss metric
     *
     * Logistic loss: y_i ln(1 + exp(-pred_i)) + (1-y_i) ln( 1 + exp(pred_i))
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
            unsigned long sz = predictions.size();
            Vector gradients(sz);

            #pragma omp parallel for schedule(static)
            for (unsigned long i = 0 ; i < sz; i++){
                gradients[i] = labels[i] - predictions[i];
            }

            return gradients;
        }

        Vector hessian(const Vector &predictions) const override {
            unsigned long sz = predictions.size();
            Vector hessians(sz);

            #pragma omp parallel for schedule(static)
            for (unsigned long i = 0 ; i < sz; i++){
                hessians[i] = abs(logit(predictions[i])) * ( 1- abs(predictions[i]));
            }

            return hessians;
        }

        double lossAt(const Vector &scores, const Vector &y) const override {
            size_t n = scores.size();
            double loss = 0.0;

            #pragma omp parallel for shared(y, scores) reduction(+: loss)
            for (size_t i = 0; i < n; i ++){
                loss += y[i] * log(clip(scores[i])) + (1 - y[i]) * log(1 - clip(scores[i]));
            }

            return -loss / n;
        }

        double scoreToPrediction(double score) const override {
            return logit(score);
        }

    };
}