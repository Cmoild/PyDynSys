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
             [](SimulationCPU& self, size_t num_points, IntegratorType iType) {
                 float* data = self.runSimulation(num_points, iType);
                 ssize_t rows = num_points, cols = 3;
                 std::vector<ssize_t> shape = {rows, cols};
                 std::vector<ssize_t> strides = {sizeof(float) * 3, sizeof(float)};
                 return py::array(py::buffer_info(
                     data,
                     sizeof(float),
                     py::format_descriptor<float>::format(),
                     2,
                     shape, strides
                 ));
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
             )pbdoc");
}

