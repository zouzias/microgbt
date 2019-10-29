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
        std::map<NodeId, std::shared_ptr<VectorB>> _leftCandidateSamples;

    public:

        explicit ClassList(long numSamples, long maxNumLeaves):
        _nodeIds(numSamples){
            _numSamples = numSamples;
            std::fill(_nodeIds.begin(), _nodeIds.end(), 0);
            for ( long i = 0; i < maxNumLeaves; i++){
                _leftCandidateSamples[i] = std::make_shared<VectorB>(_numSamples, false);
            }
        }


        void zero(NodeId nodeId) {
            auto& bitset = _leftCandidateSamples.at(nodeId);
            std::fill(bitset->begin(), bitset->end(), false);
        }

        void zero() {
            for (auto& nodeId: _leftCandidateSamples) {
                zero(nodeId.first);
            }
        }

        void erase(NodeId nodeId) {
            _leftCandidateSamples.erase(nodeId);
        }

        NodeId nodeAt(long index) const {
            return _nodeIds[index];
        }

        void appendSampleToLeftSubTree(NodeId nodeId, size_t index) {
            _leftCandidateSamples.at(nodeId)->operator[](index) = true;
        }

        void updateNodeId(long sampleIndex, NodeId newNodeId) {
            _nodeIds[sampleIndex] = newNodeId;
        }

        std::shared_ptr<VectorB> getLeft(NodeId nodeId) {
            return _leftCandidateSamples.at(nodeId);
        }


        long getLeftSize(NodeId nodeId) {
            const auto& bitset = _leftCandidateSamples.at(nodeId);
            return std::count(bitset->begin(), bitset->end(), true);
        }

        long getRightSize(NodeId nodeId) {
            return (long)_numSamples - getLeftSize(nodeId);
        }


    };
} // namespace microgbt