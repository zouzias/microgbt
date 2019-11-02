#include <histogram.h>
#include "gtest/gtest.h"

TEST(microgbt, Histogram)
{

    microgbt::VectorD featuresColumn = {1.0, 2.0, 3.0};
    microgbt::VectorD gradients = {1.0, 2.0, 3.0};
    microgbt::VectorD hessians = {1.0, 2.0, 3.0};

    microgbt::Histogram h(featuresColumn, gradients, hessians, 1);
    ASSERT_EQ(h.numBins(), 1);
}

TEST(microgbt, HistogramSingleValue)
{

    int numBins = 10;
    microgbt::VectorD featuresColumn = {1.0};
    microgbt::VectorD gradients = {1.0};
    microgbt::VectorD hessians = {1.0};

    microgbt::Histogram h(featuresColumn, gradients, hessians, numBins);

    ASSERT_EQ(h.numBins(), numBins);
    ASSERT_NEAR(h.max(), 1.0, 10e-3);
    ASSERT_NEAR(h.min(), 1.0, 10e-3);
}

TEST(microgbt, TestHistogramBinLenght)
{

    int numBins = 10;
    microgbt::VectorD featuresColumn = {1.0, 5.0};
    microgbt::VectorD gradients = {10.0, 20.0};
    microgbt::VectorD hessians = {100.0, 200.0};

    microgbt::Histogram h(featuresColumn, gradients, hessians, numBins);

    ASSERT_EQ(h.binLength(), (5.0 - 1.0) / numBins);
}

TEST(microgbt, UniformHistogram)
{

    int numBins = 5;
    microgbt::VectorD featuresColumn = {1.0, 2.0, 3.0, 4.0, 5.0};
    microgbt::VectorD gradients = {10.0, 20.0, 30.0, 40.0, 50.0};
    microgbt::VectorD hessians = {100.0, 200.0, 300.0, 400.0, 500.0};

    microgbt::Histogram h(featuresColumn, gradients, hessians, numBins);

    ASSERT_NEAR(h.max(), 5.0, 10e-3);
    ASSERT_NEAR(h.min(), 1.0, 10e-3);

    long idx = h.bin(2.0);

    ASSERT_EQ(h.gradientAtBin(idx), 20.0);

}

TEST(microgbt, UniformHistogramWithSingleValues)
{
    int numBins = 5;
    microgbt::VectorD featuresColumn = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    microgbt::VectorD gradients = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.};
    microgbt::VectorD hessians = {100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0};

    microgbt::Histogram h(featuresColumn, gradients, hessians, numBins);

    // All histogram bin must have exactly 2 values
    for (int i = 0; i < numBins; i++) {
        if ( h.lowerThreshold(i) > h.min() && h.upperThreshold(i) < h.max()) {
            ASSERT_EQ(h.getCount(i), 2);
        }
    }
}
