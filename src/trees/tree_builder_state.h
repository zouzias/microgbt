#pragma once

#include<vector>
#include<map>

namespace microgbt {

    using NodeId = long;
    using Vector = std::vector<double>;
    using GradientHessianPair = std::pair<double, double>;

    /**
    * TreeBuilderState maintains state (partial gradient / hessian sums) over tree building process
    */
    class TreeBuilderState {

        /** Maximum number of nodes */
        long _maxNumNodes;

        /** Keeps track of partial Gradient/Hessian sums for each tree node */
        std::map <NodeId, GradientHessianPair> _partialSums;

    public:
        explicit TreeBuilderState(long maxNumNodes) {
            _maxNumNodes = maxNumNodes;
        }

        void zeroAllPartialSums() {
            for( NodeId i = 0; i< _maxNumNodes; i++) {
                _partialSums[i] = GradientHessianPair{0.0, 0.0};
            }
        }

        // Get partial Gradient/Hessian sums per tree node
        inline const GradientHessianPair& partialSums(NodeId nodeId) const {
            return _partialSums.at(nodeId);
        }

        void addToPartialSums(NodeId nodeId, double g, double h) {
            const GradientHessianPair& sums = partialSums(nodeId);
            _partialSums[nodeId] = GradientHessianPair(sums.first + g, sums.second + h);
        }
    };
}
