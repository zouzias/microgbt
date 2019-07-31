#pragma once
#include <cmath>
#include "metric.h"

namespace microgbt {

    using Vector = std::vector<double>;

    class RMSE: public Metric {

    public:

        RMSE() = default;

        Vector gradients(const Vector &predictions, const Vector &labels) const override {
            Vector grads(predictions.size());

            for (size_t i = 0; i < predictions.size(); i++) {
                grads[i] = 2* (labels[i] - predictions[i]);
            }

            return grads;
        }

        Vector hessian(const Vector &scores) const override {
            // Hessian is constant vector 2.0
            return Vector(scores.size(), 2.0);

        }

        double lossAt(const Vector &scores, const Vector &y) const override {
            double loss = 0.0;

            size_t n = scores.size();
            for (size_t i = 0; i< n; i ++){
                loss += pow(y[i] - scores[i], 2.0);
            }

            return sqrt(loss / n);
        }

        double scoreToPrediction(double score) const override {
            return score;
        }
    };

} // namespace microgbt