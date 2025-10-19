#pragma once
#include <memory>
#include <try_jit.hpp>
#include <enums.hpp>
#include <array>


class SimulationCPU {
public:
    void runSimulation(size_t num_points, IntegratorType iType);

    void compileCode(std::string userCode);

public:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::unique_ptr<IntegratorCPU> integrator;
    float x_init = 1.f, y_init = 1.f, z_init = 1.f;
    float dt = 0.001f;
    float x_min, x_max, y_min, y_max, z_min, z_max;

    SimulationCPU(float deltaTime, std::array<float, 3> initConditions, float* C);
};
