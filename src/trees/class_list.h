#pragma once

namespace microgbt {

    using NodeId = long;

    /**
     * ClassList
     */
    class ClassList {
        long _numSamples;
        std::vector<NodeId> _nodeIds;

        // Node index to set of left subtree candidate samples
        std::map<NodeId, std::vector<bool>> _leftCandidateSamples;

    public:

        explicit ClassList(long numSamples, long maxNumLeaves):
        _nodeIds(numSamples){
            _numSamples = numSamples;
            std::fill(_nodeIds.begin(), _nodeIds.end(), 0);
            for ( long i = 0; i < maxNumLeaves; i++){
                _leftCandidateSamples[i] = *new std::vector<bool>(_numSamples, false);
            }
        }


        void allocateBitsets(NodeId nodeId) {
            auto& bitset = _leftCandidateSamples.at(nodeId);
            std::fill(bitset.begin(), bitset.end(), false);
        }

        void zero() {
            for (auto& nodeId: _leftCandidateSamples) {
                auto& bitset = nodeId.second;
                std::fill(bitset.begin(), bitset.end(), false);
            }
        }

        NodeId nodeAt(long index) const {
            return _nodeIds[index];
        }

        void appendSampleToLeftSubTree(NodeId nodeId, size_t index) {
            _leftCandidateSamples[nodeId][index] = true;
        }

        void updateNodeId(long sampleIndex, NodeId newNodeId) {
            _nodeIds[sampleIndex] = newNodeId;
        }

        const std::vector<bool>& getLeft(NodeId nodeId) {
            return _leftCandidateSamples.at(nodeId);
        }


        long getLeftSize(NodeId nodeId) {
            const auto& bitset = _leftCandidateSamples.at(nodeId);
            return std::count(bitset.begin(), bitset.end(), true);
        }

        long getRightSize(NodeId nodeId) {
            return (long)_numSamples - getLeftSize(nodeId);
        }


    };
} // namespace microgbt