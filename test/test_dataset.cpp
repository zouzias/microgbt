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
