#include <utils.h>
#include "gtest/gtest.h"

TEST(microgbt, Permutation)
{
    std::vector<size_t> permVec{1, 2, 0};
    microgbt::Permutation perm(permVec);


    ASSERT_EQ(perm(0), 1);
    ASSERT_EQ(perm(1), 2);
    ASSERT_EQ(perm(2), 0);

    ASSERT_EQ(perm.inverse(0), 2);
    ASSERT_EQ(perm.inverse(1), 0);
    ASSERT_EQ(perm.inverse(2), 1);
}
