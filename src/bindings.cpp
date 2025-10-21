#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <array>
#include <simulation.hpp>
#include <enums.hpp>

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
        .def(py::init([](float dt,
                         std::array<float, 3> init,
                         py::array_t<float> params) {
            if (params.ndim() != 1)
                throw std::runtime_error("params must be a 1D array");
            float* C = static_cast<float*>(params.request().ptr);
            return std::make_unique<SimulationCPU>(dt, init, C);
        }),
            py::arg("dt"),
            py::arg("init"),
            py::arg("params"),
            R"pbdoc(
                SimulationCPU(dt, init, params)
                dt - time step
                init - initial conditions [x, y, z]
                params - constants
            )pbdoc")
        .def("runSimulation",
            [](SimulationCPU& self, const size_t num_points, const IntegratorType iType) {
                float* data = self.runSimulation(num_points, iType);
                ssize_t rows = num_points, cols = 3;
                auto capsule = py::capsule(data, [](void *f) {
                    float* ptr = reinterpret_cast<float*>(f);
                    delete[] ptr;
                });
                return py::array(
                    {rows, cols},
                    {sizeof(float) * 3, sizeof(float)},
                    data,
                    capsule
                );
            },
            py::arg("steps"),
            py::arg("method"),
            R"pbdoc(
                Run simulation for given number of steps using the selected integrator.
                Returns a NumPy array of shape (steps, 3)
            )pbdoc")
        .def("compileCode", &SimulationCPU::compileCode,
            py::arg("userCode"),
            R"pbdoc(
                Compile user-defined system code at runtime.
            )pbdoc")
        .def("createOneDimBifurcationDiagram",
            [](SimulationCPU& self, const size_t num_points, const IntegratorType iType, const size_t parameterIdx, const size_t pointComponentIdx,
                                    const size_t numOfConstants,
                                    const float minValue, const float maxValue, const float deltaValue, const size_t numOfTransitionPoints) {
                auto data = self.createOneDimBifurcationDiagram(num_points, iType, parameterIdx, pointComponentIdx, numOfConstants, minValue, maxValue, deltaValue, numOfTransitionPoints);
                ssize_t rows = data->size() / 2, cols = 2;
                py::capsule owner(new std::shared_ptr<std::vector<float>>(data),
                                  [](void *p) {
                                      delete reinterpret_cast<std::shared_ptr<std::vector<float>>*>(p);
                                  });
                return py::array(
                    {rows, cols},
                    {sizeof(float) * 2, sizeof(float)},
                    data->data(),
                    owner
                );
            },
            py::arg("steps"),
            py::arg("method"),
            py::arg("parameter_idx"),
            py::arg("point_component_idx"),
            py::arg("num_of_constants"),
            py::arg("min_value"),
            py::arg("max_value"),
            py::arg("delta_value"),
            py::arg("num_transition_points"),
            R"pbdoc(
                Make a 1D bifurcation diagram
                Returns a NumPy array of shape (n, 2)
            )pbdoc");
}

