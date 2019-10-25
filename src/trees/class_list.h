#pragma once

#include <set>

namespace microgbt {

    using NodeId = long;

    /**
     * ClassList
     */
    class ClassList {

        Vector _gradients;
        Vector _hessians;
        std::vector<NodeId> _nodeIds;

        // Node index to set of left subtree candidate samples
        std::map<NodeId, std::vector<bool>> _leftCandidateSamples;

    public:

        explicit ClassList(const Vector &gradients, const Vector &hessians):
        _gradients(gradients), _hessians(hessians),
        _nodeIds(gradients.size()){
            std::fill(_nodeIds.begin(), _nodeIds.end(), 0);
        }


        void initBitSets(NodeId nodeId) {
            if (_leftCandidateSamples.find(nodeId) == _leftCandidateSamples.end()) {

                _leftCandidateSamples[nodeId] = *new std::vector<bool>(_gradients.size(), false);
            } else {
                std::fill(_leftCandidateSamples[nodeId].begin(), _leftCandidateSamples[nodeId].end(), false);
            }
        }

        NodeId nodeAt(long index) const {
            return _nodeIds[index];
        }

        void appendSampleToLeftSubTree(NodeId nodeId, long index) {
            _leftCandidateSamples[nodeId][index] = true;
        }

        void updateNodeId(long sampleIndex, NodeId newNodeId) {
            _nodeIds[sampleIndex] = newNodeId;
        }

        const std::vector<bool>& getLeft(NodeId nodeId) {
            return _leftCandidateSamples.at(nodeId);
        }

        inline long getLeftSize(NodeId nodeId) const {
            return _leftCandidateSamples.at(nodeId).size();
        }

        inline long getRightSize(NodeId nodeId) const {
            return (long)_gradients.size() - getLeftSize(nodeId);
        }

    };
} // namespace microgbt