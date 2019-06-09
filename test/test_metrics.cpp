#include <metrics/logloss.h>
#include <metrics/rmse.h>
#include "gtest/gtest.h"

TEST(microgbt, LogLoss)
{
    microgbt::LogLoss logloss;
    ASSERT_NEAR(logloss.logit(1.0), 1 / ( 1 + exp(-1.0)), 1.0e-11);
}

TEST(microgbt, RMSE)
{
    microgbt::RMSE rmse;
    ASSERT_NEAR(rmse.scoreToPrediction(10.1), 10.1, 1.0e-11);
}
