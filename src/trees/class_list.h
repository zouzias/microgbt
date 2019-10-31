#pragma once
#include <memory>

namespace microgbt {

    using NodeId = long;
    using VectorB = std::vector<bool>;

    /**
     * ClassList
     */
    class ClassList {
        long _numSamples;
        std::vector<NodeId> _nodeIds;

        // Node index to set of left subtree candidate samples
        std::map<NodeId, long> _leftCandidateSamples;

    public:

        explicit ClassList(long numSamples):
        _nodeIds(numSamples){
            _numSamples = numSamples;
            std::fill(_nodeIds.begin(), _nodeIds.end(), 0);
        }


        void zero(NodeId nodeId) {
            _leftCandidateSamples[nodeId] = 0;
        }

        void zero() {
            for (auto& node: _leftCandidateSamples) {
                node.second = 0;
            }
        }

        NodeId nodeAt(long index) const {
            return _nodeIds[index];
        }

        void increaseLeftSizeByNode(NodeId nodeId) {
            _leftCandidateSamples[nodeId] ++;
        }

        void updateNodeId(long sampleIndex, NodeId newNodeId) {
            _nodeIds[sampleIndex] = newNodeId;
        }

        /**
         * Returns the candidate left sub-tree size for a node
         * @param nodeId Node id
         * @return
         */
        long getLeftSize(NodeId nodeId) {
            return _leftCandidateSamples[nodeId];
        }

        long getRightSize(NodeId nodeId) {
            return (long)_numSamples - getLeftSize(nodeId);
        }


    };
} // namespace microgbt