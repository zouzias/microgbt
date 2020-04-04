add_library(microgbt STATIC metrics/metric.h trees/tree.h trees/split_info.h
        GBT.h dataset.h metrics/logloss.h trees/treenode.h metrics/rmse.h
        trees/numerical_splliter.h trees/splitter.h types.h)
set_target_properties(microgbt PROPERTIES POSITION_INDEPENDENT_CODE ON)



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
