#include <dataset.h>
#include "gtest/gtest.h"

TEST(Dataset, DefaultConstructor)
{

    size_t m = 2, n = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
    microgbt::Vector y = {1.0, 2.0};
    microgbt::Dataset dataset(A, y);

    ASSERT_EQ(dataset.nRows(),  m);
    ASSERT_EQ(dataset.numFeatures(), n);
}

TEST(Dataset, Constructor)
{

    int m = 2, n = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
    microgbt::Vector y = {1.0, 2.0};
    microgbt::Dataset dataset(A, y);

    std::vector<size_t> left = {0};
    std::vector<size_t> right = {1, 2};
    microgbt::SplitInfo splitInfo(0.0, 1.0, left, right, left, right);

    microgbt::Dataset leftDS(dataset, splitInfo, microgbt::SplitInfo::Left);

    ASSERT_EQ(leftDS.nRows(), left.size());
    ASSERT_EQ(leftDS.numFeatures(), n);

}
