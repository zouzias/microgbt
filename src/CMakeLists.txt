add_library(microgbt "" metrics/metric.h trees/tree.h
        GBT.h dataset.h metrics/logloss.h trees/treenode.h metrics/rmse.h
        utils.h trees/class_list.h trees/tree_builder_state.h)


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
