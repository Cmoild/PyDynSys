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
        .def(py::init<float, std::array<float,3>, float*>())
        .def("runSimulation", &SimulationCPU::runSimulation,
             py::arg("steps"), py::arg("method"))
        .def("compileCode", &SimulationCPU::compileCode,
             py::arg("userCode"));
}
