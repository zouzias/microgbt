#include "gtest/gtest.h"
#include "trees/split_info.h"

TEST(microgbt, Gain)
{
    microgbt::SplitInfo splitInfo(0.0, 1.0);
    ASSERT_NEAR(splitInfo.bestGain(), 0.0, 1.0e-11);
    ASSERT_NEAR(splitInfo.splitValue(), 1.0, 1.0e-11);
}
