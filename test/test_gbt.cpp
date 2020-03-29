#include <GBT.h>
#include "gtest/gtest.h"

TEST(GBT, LAMBDA)
{

        const std::map<std::string, double> params{
            {"lambda", 11.0},
            {"gamma", 12.0},
            {"shrinkage_rate", 13.0},
            {"min_split_gain", 14.0},
            {"min_tree_size", 15.0},
            {"learning_rate", 16.0},
            {"min_split_gain", 17.0},
            {"max_depth", 18.0},
            {"metric", 19.0}};

        microgbt::GBT gbt(params);

        ASSERT_EQ(gbt.lambda(), 11.0);
}

TEST(GBT, MINSPLITGAIN)
{

        const std::map<std::string, double> params{
            {"lambda", 11.0},
            {"gamma", 12.0},
            {"shrinkage_rate", 13.0},
            {"min_split_gain", 14.0},
            {"min_tree_size", 15.0},
            {"learning_rate", 16.0},
            {"min_split_gain", 17.0},
            {"max_depth", 18.0},
            {"metric", 19.0}};

        microgbt::GBT gbt(params);

        ASSERT_EQ(gbt.minSplitGain(), 14.0);
}

TEST(GBT, LEARNINGRATE)
{

        const std::map<std::string, double> params{
            {"lambda", 11.0},
            {"gamma", 12.0},
            {"shrinkage_rate", 13.0},
            {"min_split_gain", 14.0},
            {"min_tree_size", 15.0},
            {"learning_rate", 16.0},
            {"min_split_gain", 17.0},
            {"max_depth", 18.0},
            {"metric", 19.0}};

        microgbt::GBT gbt(params);

        ASSERT_EQ(gbt.getLearningRate(), 16.0);
}

TEST(GBT, MAXDEPTH)
{

        const std::map<std::string, double> params{
            {"lambda", 11.0},
            {"gamma", 12.0},
            {"shrinkage_rate", 13.0},
            {"min_split_gain", 14.0},
            {"min_tree_size", 15.0},
            {"learning_rate", 16.0},
            {"min_split_gain", 17.0},
            {"max_depth", 18.0},
            {"metric", 19.0}};

        microgbt::GBT gbt(params);

        ASSERT_EQ(gbt.maxDepth(), 18.0);
}
