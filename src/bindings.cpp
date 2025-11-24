#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <array>
#include <simulation.hpp>
#include <enums.hpp>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(_dynsys, m) {
    m.doc() = "Bindings for dynamics simulation backend";

    py::enum_<IntegratorType>(m, "IntegratorType")
        .value("RUNGE_KUTTA_4", RUNGE_KUTTA_4)
        .value("EULER", EULER)
        .value("EULER_CROMER", EULER_CROMER)
        .value("MIDPOINT", MIDPOINT)
        .value("CD", CD_METOD)
        .export_values();

    py::class_<SimulationCPU>(m, "SimulationCPU")
        .def(py::init([](float dt, std::array<float, 3> init, py::array_t<float> params) {
                 if (params.ndim() != 1)
                     throw std::runtime_error("params must be a 1D array");
                 float* C = static_cast<float*>(params.request().ptr);
                 return std::make_unique<SimulationCPU>(dt, init, C, params.size());
             }),
             py::arg("dt"), py::arg("init"), py::arg("params"),
             R"pbdoc(
                SimulationCPU(dt, init, params)
                dt - time step
                init - initial conditions [x, y, z]
                params - constants
            )pbdoc")
        .def(
            "runSimulation",
            [](SimulationCPU& self, const size_t num_points, const IntegratorType iType) {
                float* data = self.runSimulation(num_points, iType);
                ssize_t rows = num_points, cols = 3;
                auto capsule = py::capsule(data, [](void* f) {
                    float* ptr = reinterpret_cast<float*>(f);
                    delete[] ptr;
                });
                return py::array(py::dtype::of<float>(), {rows, cols},
                                 {sizeof(float) * 3, sizeof(float)}, data, capsule);
            },
            py::arg("steps"), py::arg("method"),
            R"pbdoc(
                Run simulation for given number of steps using the selected integrator.
                Returns a NumPy array of shape (steps, 3)
            )pbdoc")
        .def("compileCode", &SimulationCPU::compileCode, py::arg("userCode"),
             R"pbdoc(
                Compile user-defined system code at runtime.
            )pbdoc")
        .def(
            "createOneDimBifurcationDiagram",
            [](SimulationCPU& self, const size_t num_points, const IntegratorType iType,
               const size_t parameterIdx, const size_t pointComponentIdx,
               const size_t numOfConstants, const float minValue, const float maxValue,
               const float deltaValue, const size_t numOfTransitionPoints) {
                auto data = self.createOneDimBifurcationDiagram(
                    num_points, iType, parameterIdx, pointComponentIdx, numOfConstants, minValue,
                    maxValue, deltaValue, numOfTransitionPoints);
                ssize_t rows = data->size() / 3, cols = 3;
                // py::capsule owner(new std::shared_ptr<std::vector<float>>(data), [](void* p) {
                //     delete reinterpret_cast<std::shared_ptr<std::vector<float>>*>(p);
                // });
                return py::array(py::dtype::of<float>(), {rows, cols},
                                 {sizeof(float) * 3, sizeof(float)}, data->data() /*, owner */);
            },
            py::arg("steps"), py::arg("method"), py::arg("parameter_idx"),
            py::arg("point_component_idx"), py::arg("num_of_constants"), py::arg("min_value"),
            py::arg("max_value"), py::arg("delta_value"), py::arg("num_transition_points"),
            R"pbdoc(
                Make a 1D bifurcation diagram
                Returns a NumPy array of shape (n, 3)
                Row: [cur_constant, peak_val, interval_val]
            )pbdoc")
        .def(
            "createTwoDimBifurcationDiagram",
            [](SimulationCPU& self, const size_t num_points, const IntegratorType iType,
               const size_t parameterIdx1, const size_t parameterIdx2,
               const size_t pointComponentIdx, const size_t numOfConstants, const float minValue1,
               const float maxValue1, const float deltaValue1, const float minValue2,
               const float maxValue2, const float deltaValue2, const size_t numOfTransitionPoints) {
                auto data = self.createTwoDimBifurcatonDiagram(
                    num_points, iType, parameterIdx1, parameterIdx2, pointComponentIdx,
                    numOfConstants, minValue1, maxValue1, deltaValue1, minValue2, maxValue2,
                    deltaValue2, numOfTransitionPoints);
                size_t rows = (size_t)((maxValue2 - minValue2) / deltaValue2) + 1;
                size_t cols = (size_t)((maxValue1 - minValue1) / deltaValue1) + 1;
                // py::capsule owner(new std::shared_ptr<std::vector<float>>(data), [](void* p) {
                //     delete reinterpret_cast<std::shared_ptr<std::vector<float>>*>(p);
                // });
                return py::array(py::dtype::of<float>(), {rows, cols, static_cast<size_t>(1)},
                                 {sizeof(float) * cols, sizeof(float), sizeof(float)}, data->data() /*,
                                 owner */);
            },
            py::arg("steps"), py::arg("method"), py::arg("parameter_idx1"),
            py::arg("parameter_idx2"), py::arg("point_component_idx"), py::arg("num_of_constants"),
            py::arg("min_value1"), py::arg("max_value1"), py::arg("delta_value1"),
            py::arg("min_value2"), py::arg("max_value2"), py::arg("delta_value2"),
            py::arg("num_transition_points"),
            R"pbdoc(
                Make a 2D bifurcation diagram
                Returns a NumPy array of shape (num_first_constant_vals, num_second_constant_vals, 1)
            )pbdoc")
        .def(
            "createOneDimLyapunovDiagram",
            [](SimulationCPU& self, const size_t num_points, const IntegratorType iType,
               const size_t parameterIdx, const size_t numOfConstants, const float minValue,
               const float maxValue, const float deltaValue, const std::array<size_t, 3> xyzOrder,
               std::string varSysCode) {
                auto data = self.createOneDimLyapunovDiagram(num_points, iType, parameterIdx,
                                                             numOfConstants, minValue, maxValue,
                                                             deltaValue, xyzOrder, varSysCode);
                ssize_t rows = data->size() / 2, cols = 2;
                return py::array(py::dtype::of<float>(), {rows, cols},
                                 {sizeof(float) * 2, sizeof(float)}, data->data());
            },
            py::arg("steps"), py::arg("method"), py::arg("parameter_idx"),
            py::arg("num_of_constants"), py::arg("min_value"), py::arg("max_value"),
            py::arg("delta_value"), py::arg("xyz_order"), py::arg("var_sys_code"),
            R"pbdoc(
                Make a 1D lyapunov diagram
                Returns a NumPy array of shape (n, 2)
            )pbdoc")
        .def(
            "createTwoDimLyapunovDiagram",
            [](SimulationCPU& self, const size_t num_points, const IntegratorType iType,
               const size_t parameterIdx1, const size_t parameterIdx2, const size_t numOfConstants,
               const float minValue1, const float maxValue1, const float deltaValue1,
               const float minValue2, const float maxValue2, const float deltaValue2,
               const std::array<size_t, 3> xyzOrder, std::string varSysCode) {
                auto data = self.createTwoDimLyapunovDiagram(
                    num_points, iType, parameterIdx1, parameterIdx2, numOfConstants, minValue1,
                    maxValue1, deltaValue1, minValue2, maxValue2, deltaValue2, xyzOrder,
                    varSysCode);
                size_t rows = (size_t)((maxValue2 - minValue2) / deltaValue2) + 1;
                size_t cols = (size_t)((maxValue1 - minValue1) / deltaValue1) + 1;
                return py::array(py::dtype::of<float>(), {rows, cols, static_cast<size_t>(1)},
                                 {sizeof(float) * cols, sizeof(float), sizeof(float)},
                                 data->data());
            },
            py::arg("steps"), py::arg("method"), py::arg("parameter_idx1"),
            py::arg("parameter_idx2"), py::arg("num_of_constants"), py::arg("min_value1"),
            py::arg("max_value1"), py::arg("delta_value1"), py::arg("min_value2"),
            py::arg("max_value2"), py::arg("delta_value2"), py::arg("xyz_order"),
            py::arg("var_sys_code"),
            R"pbdoc(
                Make a 2D Lyapunov diagram
                Returns a NumPy array of shape (num_first_constant_vals, num_second_constant_vals, 1)
            )pbdoc");
}
