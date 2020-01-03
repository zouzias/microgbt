#include <metrics/logloss.h>
#include <metrics/rmse.h>
#include "gtest/gtest.h"

using namespace microgbt;

TEST(microgbt, LogLoss)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.logit(1.0), 1 / ( 1 + exp(-1.0)), 1.0e-11);
}

TEST(microgbt, LogLossMiddleValue)
{
    LogLoss logloss;
    ASSERT_NEAR(logloss.logit(0.0), 0.5, 1.0e-11);
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

TEST(microgbt, LogLossGradient)
{
    LogLoss logloss;
    Vector preds = Vector::Constant(10, 100.0);
    Vector targets = Vector::Constant(10, 99.0);

    Vector grads = logloss.gradients(preds, targets);
    ASSERT_EQ(grads.size(), preds.size());
    ASSERT_NEAR(grads[0], 100.0 - 99, 1.0e-7);
}

TEST(microgbt, LogLossHessian)
{
    LogLoss logloss;
    Vector preds = Vector::Constant(10, 0.5);

    Vector hessian = logloss.hessian(preds);
    ASSERT_EQ(hessian.size(), preds.size());
    ASSERT_NEAR(hessian[0], 0.25, 1.0e-7);
}

TEST(microgbt, LogLossLossAtMustBeZero)
{
    LogLoss logloss;
    Vector preds = Vector::Constant(10, 1.0);
    Vector targets = Vector::Constant(10, 1.0);

    double loss = logloss.lossAt(preds, targets);
    ASSERT_NEAR(loss, 0, 1.0e-7);
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
    Vector preds = Vector::Constant(10, 100.0);
    Vector targets = Vector::Constant(10, 99.0);

    Vector grads = rmse.gradients(preds, targets);
    ASSERT_EQ(grads.size(), preds.size());
    ASSERT_NEAR(grads[0], 2 * (100.0 - 99), 1.0e-7);
}

TEST(microgbt, RMSELossAtMustBeZero)
{
    RMSE rmse;

    Vector preds = Vector::Constant(10, 1.0);
    Vector targets = Vector::Constant(10, 1.0);

    double loss = rmse.lossAt(preds, targets);
    ASSERT_NEAR(loss, 0, 1.0e-7);
}
