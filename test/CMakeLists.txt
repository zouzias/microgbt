add_executable(
    unit_tests
        main.cpp
        test_split_info.cpp
        test_metric_logloss.cpp
        test_metric_rmse.cpp
        test_dataset.cpp
        test_treenode.cpp
        test_tree.cpp
        test_gbt.cpp
    ../src/metrics/metric.h ../src/trees/tree.h ../src/GBT.h ../src/dataset.h ../src/metrics/rmse.h)

target_link_libraries(
    unit_tests
    gtest_main
    microgbt
    OpenMP::OpenMP_CXX
)

add_test(
  NAME
    unit
  COMMAND
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests
  )
