#pragma once
#include <cmath>
#include "metric.h"

namespace microgbt {

    class RMSE: public Metric {

    public:

        RMSE() = default;

        Vector gradients(const Vector &predictions, const Vector &labels) const override {
            return 2 * (predictions - labels);
        }

        Vector hessian(const Vector &predictions) const override {
            // Hessian is constant vector 2.0
            return Eigen::VectorXf::Constant(predictions.size(), 2.0);

        }

        double lossAt(const Vector &predictions, const Vector &labels) const override {
            long double loss = 0.0;

            long n = predictions.size();
            for (long i = 0; i< n; i ++){
                loss += pow(labels[i] - predictions[i], 2.0);
            }

            return (double)std::sqrt(loss / n);
        }

        double scoreToPrediction(double score) const override {
            return score;
        }
    };

} // namespace microgbt