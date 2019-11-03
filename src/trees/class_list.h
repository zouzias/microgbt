#pragma once
#include <memory>

namespace microgbt {

    using NodeId = long;
    using VectorB = std::vector<bool>;

    /**
     * ClassList that keeps track of left subtree sizes over all active tree nodes
     *
     * active is a tree node that could be split further
     */
    class ClassList {

        // Number of samples
        long _numSamples;

        // Vector of tree node ids
        std::vector<NodeId> _nodeIds;

        // Keep track of size of left subtree for a given tree node (NodeId)
        // Map from node index -> size of left subtree of given index
        std::map<NodeId, long> _leftCandidateSamples;

    public:

        explicit ClassList(long numSamples): _nodeIds(numSamples){
            // All samples are assigned to root node
            std::fill(_nodeIds.begin(), _nodeIds.end(), 0);
            _numSamples = numSamples;
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
         * Returns the candidate left sub-tree size of a tree node
         */
        long getLeftSize(NodeId nodeId) {
            return _leftCandidateSamples[nodeId];
        }

        long getRightSize(NodeId nodeId) {
            return (long)_numSamples - getLeftSize(nodeId);
        }
    };
} // namespace microgbt