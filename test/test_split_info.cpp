#include "gtest/gtest.h"
#include "trees/split_info.h"

TEST(microgbt, SplitInfo)
{
    microgbt::SplitInfo gain(0.0, 1.0);
    ASSERT_NEAR(gain.bestGain(), 0.0, 1.0e-11);
}

TEST(microgbt, SplitInfoSplitValue)
{
    microgbt::SplitInfo gain(0.0, 1.0);
    ASSERT_NEAR(gain.splitValue(), 1.0, 1.0e-11);
}
