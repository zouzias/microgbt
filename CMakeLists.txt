cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

PROJECT(microgbt CXX)

OPTION(USE_OPENMP "Enable OpenMP" ON)

# project version
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)


# enable c++ language
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Look for eigen 3.3
FIND_PACKAGE (Eigen3 3.3 REQUIRED NO_MODULE)



# Based on https://iscinumpy.gitlab.io/post/omp-on-high-sierra/
# and for Mojave 10.14: see https://github.com/microsoft/LightGBM/pull/1919
if(USE_OPENMP)
    find_package(OpenMP)

    # If OpenMP wasn't found, try if we can find it in the default Homebrew location
    if((NOT OPENMP_FOUND) AND (NOT OPENMP_CXX_FOUND) AND EXISTS "/usr/local/opt/libomp/lib/libomp.dylib")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES omp)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)

        find_package(OpenMP)
        if (OPENMP_FOUND OR OPENMP_CXX_FOUND)
            message(STATUS "Found libomp in homebrew default location.")
        else()
            message(FATAL_ERROR "Didn't find libomp. Tried homebrew default location but also didn't find it.")
        endif()
    endif()

    if((NOT OPENMP_FOUND) AND (NOT OPENMP_CXX_FOUND))
        message(FATAL_ERROR "Did not find OpenMP.")
    endif()

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    # Ignore unknown #pragma warning
    if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif(USE_OPENMP)

# place binaries and libraries according to GNU standards
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# we use this to get code coverage
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

if(UNIX OR MINGW OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -march=native -Ofast -Wextra -Wall -Wno-ignored-attributes -Wno-unknown-pragmas -Wno-return-type")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
endif()

add_subdirectory(src)


# ------------------------------------------------------------------------------
# Valgrind
# ------------------------------------------------------------------------------
include (CTest)
set(MemoryCheckCommand /usr/bin/valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --leak-check=full")
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --track-fds=yes")
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --trace-children=yes")
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --track-origins=yes")
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --error-exitcode=1")

include(cmake/googletest.cmake)
fetch_googletest(
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/googletest
    )

enable_testing()
add_subdirectory(test)


#########################
# pybind11 integration  #
#########################
find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

# For Python integration
add_subdirectory(pybind11)

pybind11_add_module(microgbtpy src/metrics/metric.h src/trees/tree.h src/GBT.h src/dataset.h src/metrics/logloss.h
        src/trees/treenode.h src/trees/split_info.h
        src/utils.h
        src/python_api.cpp)
#########################


#########################
# Link openMP
#########################
if(USE_OPENMP)
    target_link_libraries(microgbtpy PRIVATE OpenMP::OpenMP_CXX)
endif(USE_OPENMP)
#########################
