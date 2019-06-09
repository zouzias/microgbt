add_library(microgbt "" metrics/metric.h trees/tree.h GBT.h dataset.h metrics/logloss.h trees/treenode.h trees/split_info.h metrics/rmse.h)


target_sources(
    microgbt
    PRIVATE
        main.cpp
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/dataset.h
    )

target_include_directories(
    microgbt
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
    )

target_link_libraries (microgbt Eigen3::Eigen)
