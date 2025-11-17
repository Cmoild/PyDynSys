#pragma once
#include <cstddef>
#include <memory>
#include <try_jit.hpp>
#include <enums.hpp>
#include <array>
#include <vector>

class SimulationCPU {
  public:
    float* runSimulation(const size_t num_points, const IntegratorType iType);

    void compileCode(std::string userCode);

    std::shared_ptr<std::vector<float>> createOneDimBifurcationDiagram(
        const size_t num_points, const IntegratorType iType, const size_t parameterIdx,
        const size_t pointComponentIdx, const size_t numOfConstants, const float minValue,
        const float maxValue, const float deltaValue, const size_t numOfTransitionPoints);

    std::shared_ptr<std::vector<float>> createTwoDimBifurcatonDiagram(
        const size_t num_points, const IntegratorType iType, const size_t parameterIdx1,
        const size_t parameterIdx2, const size_t pointComponentIdx, const size_t numOfConstants,
        const float minValue1, const float maxValue1, const float deltaValue1,
        const float minValue2, const float maxValue2, const float deltaValue2,
        const size_t numOfTransitionPoints);

    std::shared_ptr<std::vector<float>>
    createOneDimLyapunovDiagram(const size_t num_points, const IntegratorType iType,
                                const size_t parameterIdx, const size_t numOfConstants,
                                const float minValue, const float maxValue, const float deltaValue,
                                const std::array<size_t, 3> xyzOrder,
                                const std::array<size_t, 4> constOrder);

  public:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::unique_ptr<IntegratorCPU> integrator;
    float x_init = 1.f, y_init = 1.f, z_init = 1.f;
    float dt = 0.001f;
    float x_min, x_max, y_min, y_max, z_min, z_max;
    std::vector<float> constants;

    SimulationCPU(float deltaTime, std::array<float, 3> initConditions, float* C, size_t numConsts);
};
