#pragma once

#include "../dataset.h"
#include "split_info.h"

namespace microgbt {

    /**
     * Splitter defines a binary tree splits interface
     */
    class Splitter {
    public:

        virtual ~Splitter() = default;

        /**
         * Return the best binary tree split based on a dataset (matrix, target vector) and
         * the corresponding gradient and Hessian vectors
         *
         * @param dataset Current dataset (matrix, train vector)
         * @return Best split over all features and all samples
s         */
        virtual SplitInfo findBestSplit(const Dataset& dataset) const = 0;
    };
}