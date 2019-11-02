#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>
#include "GBT.h"

namespace py = pybind11;

PYBIND11_MODULE(microgbtpy, m)
{
        m.doc() = "microGBT Python API";

        py::class_<microgbt::GBT> gbt(m, "GBT");

// Common methods
gbt.def(py::init<std::map<std::string, double> >())
        .def("max_depth", &microgbt::GBT::maxDepth)
        .def("min_split_gain", &microgbt::GBT::minSplitGain)
        .def("learning_rate", &microgbt::GBT::getLearningRate)
        .def("get_lambda", &microgbt::GBT::lambda)
        .def("max_bin", &microgbt::GBT::maxHistogramBin)
        .def("best_iteration", &microgbt::GBT::getBestIteration);

        // Train API
        gbt.def("train", &microgbt::GBT::trainPython,
                "Python API for microGBT training",
                py::call_guard<py::gil_scoped_release>(),
                pybind11::arg("train_X"), pybind11::arg("train_y"),
                pybind11::arg("valid_x"), pybind11::arg("valid_y"),
                pybind11::arg("num_iterations"), pybind11::arg("early_stopping_rounds") = 5);

        // Predict API
        gbt.def("predict", &microgbt::GBT::predict, "Python API to get predictions using microGBT",
                pybind11::arg("x"),
                pybind11::arg("num_iterations") = 0);

gbt.def("__repr__",
                 [](const microgbt::GBT &a) {
                     std::string repr;
                     repr += "<microgbt>[";
                     repr += "learningRate:";
                     repr += std::to_string(a.getLearningRate());
                     repr += ",maxDepth:";
                     repr += std::to_string(a.maxDepth());
                     repr += ",shrinkageRate:";
                     repr += std::to_string(a.shrinkageRate());
                     repr += ",minSplitGain:";
                     repr += std::to_string(a.minSplitGain());
                     repr += ",lambda:";
                     repr += std::to_string(a.lambda());
                     repr += ",max_bin:";
                     repr += std::to_string(a.maxHistogramBin());
                     repr += "]";
                     return repr;
                 }
            );
} // PYBIND11_MODULE
