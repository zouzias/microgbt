#include <dataset.h>
#include "gtest/gtest.h"

TEST(Dataset, DefaultConstructor)
{

    long m = 2, n = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
    microgbt::Vector y = {1.0, 2.0};
    microgbt::Dataset dataset(A, y);

    ASSERT_EQ(dataset.nRows(),  m);
    ASSERT_EQ(dataset.numFeatures(), n);
}

TEST(Dataset, Coeff)
{

    long m = 2, n = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
    microgbt::Vector y = {1.0, 2.0};
    microgbt::Dataset dataset(A, y);

    ASSERT_EQ(dataset.coeff(1,1), 0.0);
}
