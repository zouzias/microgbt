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
        std::map<NodeId, std::set<long>> _leftCandidateSamples;

    public:

        explicit ClassList(const Vector &gradients, const Vector &hessians):
        _gradients(gradients), _hessians(hessians),
        _nodeIds(gradients.size()){
            std::fill(_nodeIds.begin(), _nodeIds.end(), 0);
        }

        void clean() {
            _leftCandidateSamples.clear();
        }

        NodeId nodeAt(long index) const {
            return _nodeIds[index];
        }

        void appendSampleToLeftSubTree(NodeId nodeId, long index) {
            _leftCandidateSamples[nodeId].insert(index);
        }

        void updateNodeId(long sampleIndex, NodeId newNodeId) {
            _nodeIds[sampleIndex] = newNodeId;
        }

        std::set<long> getLeft(NodeId nodeId) {
            return _leftCandidateSamples[nodeId];
        }

        long getLeftSize(NodeId nodeId) {
            return _leftCandidateSamples[nodeId].size();
        }

        long getRightSize(NodeId nodeId) {
            return (long)_gradients.size() - (long)_leftCandidateSamples[nodeId].size();
        }

    };
} // namespace microgbt