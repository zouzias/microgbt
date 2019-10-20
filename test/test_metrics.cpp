#include <metrics/logloss.h>
#include <metrics/rmse.h>
#include "gtest/gtest.h"

using namespace microgbt;

TEST(microgbt, LogLoss)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.logit(1.0), 1 / ( 1 + exp(-1.0)), 1.0e-11);
}

TEST(microgbt, LogLossClipUpper)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.clip(1.0), 1, 1.0e-7);
}

TEST(microgbt, LogLossClipUpperOverflow)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.clip(100.0), 1, 1.0e-7);
}

TEST(microgbt, LogLossClipLower)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.clip(0.0), 0, 1.0e-7);
}

TEST(microgbt, LogLossClipLowerUnderFlow)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.clip(-10000.0), 0, 1.0e-7);
}

TEST(microgbt, RMSE)
{
    RMSE rmse;
    ASSERT_NEAR(rmse.scoreToPrediction(10.1), 10.1, 1.0e-11);
}

TEST(microgbt, RMSEHessian)
{
    RMSE rmse;
    Vector preds = Vector(10);
    Vector hessian = rmse.hessian(preds);
    ASSERT_EQ(hessian.size(), preds.size());

    // Hessian is the constant 2
    ASSERT_NEAR(hessian[0], 2.0, 1.0e-11);
    ASSERT_NEAR(hessian[9], 2.0, 1.0e-11);
}

TEST(microgbt, RMSEGradient)
{
    RMSE rmse;
    Vector preds = Vector(10);
    Vector targets = Vector(10);

    std::fill(preds.begin(), preds.end(), 100.0);
    std::fill(targets.begin(), targets.end(), 99.0);

    Vector grads = rmse.gradients(preds, targets);
    ASSERT_EQ(grads.size(), preds.size());
    ASSERT_NEAR(grads[0], 2 * (100.0 - 99), 1.0e-7);
}
