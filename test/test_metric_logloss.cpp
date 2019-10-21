#include <metrics/logloss.h>
#include <metrics/rmse.h>
#include "gtest/gtest.h"

using namespace microgbt;

TEST(LogLoss, LogLossLogit)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.logit(1.0), 1 / ( 1 + exp(-1.0)), 1.0e-11);
}

TEST(LogLoss, LogLossMiddleValue)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.logit(0.0), 0.5, 1.0e-11);
}

TEST(LogLoss, LogLossClipUpper)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.clip(1.0), 1, 1.0e-7);
}

TEST(LogLoss, LogLossClipUpperOverflow)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.clip(100.0), 1, 1.0e-7);
}

TEST(LogLoss, LogLossClipLower)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.clip(0.0), 0, 1.0e-7);
}

TEST(LogLoss, LogLossClipLowerUnderFlow)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.clip(-10000.0), 0, 1.0e-7);
}

TEST(LogLoss, LogLossGradient)
{
    LogLoss logloss;
    Vector preds = Vector(10);
    Vector targets = Vector(10);

    std::fill(preds.begin(), preds.end(), 100.0);
    std::fill(targets.begin(), targets.end(), 99.0);

    Vector grads = logloss.gradients(preds, targets);
    ASSERT_EQ(grads.size(), preds.size());
    ASSERT_NEAR(grads[0], 100.0 - 99, 1.0e-7);
}

TEST(LogLoss, LogLossHessian)
{
    LogLoss logloss;
    Vector preds = Vector(10);

    std::fill(preds.begin(), preds.end(), 0.5);

    Vector hessian = logloss.hessian(preds);
    ASSERT_EQ(hessian.size(), preds.size());
    ASSERT_NEAR(hessian[0], 0.25, 1.0e-7);
}

TEST(LogLoss, LogLossLossAtMustBeZero)
{
    LogLoss logloss;
    Vector preds = Vector(10);
    Vector targets = Vector(10);

    std::fill(preds.begin(), preds.end(), 1.0);
    std::fill(targets.begin(), targets.end(), 1.0);

    double loss = logloss.lossAt(preds, targets);
    ASSERT_NEAR(loss, 0, 1.0e-7);
}