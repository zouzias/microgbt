#pragma once

#include "../dataset.h"
#include "split_info.h"

namespace microgbt {

    /**
     * Splitter defines the interface for binary tree splits
     *
     */
    class Splitter {

    public:

        virtual ~Splitter() = default;

        /**
         * Return split information for a given dataset (matrix, target vector) and
         * the corresponding gradient and Hessian vectors
         *
         * @param dataset Current dataset (matrix, train vector)
         * @param gradient Gradient vector, one coordinate per sample / dataset row
         * @param hessian Hessian vector, one coordinate per sample / dataset row
         * @return Best split over all features and all samples
         */
        virtual SplitInfo findBestSplit(const Dataset &dataset,
                                        const Vector &gradient,
                                        const Vector &hessian) const = 0;
    };
} // namespace microgbt