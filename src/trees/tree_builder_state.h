#pragma once

#include<vector>
#include<map>

namespace microgbt {

    using NodeId = long;
    using Vector = std::vector<double>;
    using GradientHessianPair = std::pair<double, double>;

    /**
    * Maintains state (partial gradient / hessian sums) over tree building process
    */
    class TreeBuilderState {
        std::shared_ptr <Vector> _gradient;
        std::shared_ptr <Vector> _hessian;
        std::map <NodeId, GradientHessianPair> _partialSums;


    public:
        TreeBuilderState(const Vector &gradient, const Vector &hessian) : _gradient(std::make_shared<Vector>(gradient)),
                                                                          _hessian(std::make_shared<Vector>(hessian)),
                                                                          _partialSums() {
        }

        // Get partial Gradient/Hessian sums per tree node
        GradientHessianPair partialSums(NodeId nodeId) const {
            if (_partialSums.find(nodeId) == _partialSums.end()){
                return GradientHessianPair{0.0, 0.0};
            }
            else {
                return _partialSums.at(nodeId);
            }
        }

        void addToPartialSums(NodeId nodeId, double g, double h) {
            GradientHessianPair sums = partialSums(nodeId);
            _partialSums[nodeId] = GradientHessianPair(sums.first + g, sums.second + h);
        }
    };
}
