#include <trees/treenode.h>
#include "gtest/gtest.h"

TEST(microgbt, TreeNodeConstructor)
{
    microgbt::TreeNode treeNode(1.0, 2.0, 10, 0);
    ASSERT_TRUE(true);
}
