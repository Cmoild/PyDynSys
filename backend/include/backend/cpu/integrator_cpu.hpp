#pragma once
#include <memory>
#include <stddef.h>

typedef void (*integratedFunc)(const float* X, const float* C, float* X_dot);

struct IntegratorCPU {
    float step;
    float* points;
    float* C;
    size_t cur_step;
    integratedFunc func;
};

void integrator_euler_step(std::unique_ptr<IntegratorCPU>& integrator);

void integrator_rk4_step(std::unique_ptr<IntegratorCPU>& integrator);

void integrator_midpoint_step(std::unique_ptr<IntegratorCPU>& integrator);

void integrator_euler_cromer_step(std::unique_ptr<IntegratorCPU>& integrator);

void integrator_cd_step(std::unique_ptr<IntegratorCPU>& integrator);
