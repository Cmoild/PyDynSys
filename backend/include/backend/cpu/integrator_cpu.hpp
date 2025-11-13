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

static_assert(offsetof(IntegratorCPU, points) == 8, "offset mismatch");
static_assert(offsetof(IntegratorCPU, C) == 16, "offset mismatch");
static_assert(offsetof(IntegratorCPU, cur_step) == 24, "offset mismatch");
static_assert(offsetof(IntegratorCPU, func) == 32, "offset mismatch");

void integrator_euler_step(std::unique_ptr<IntegratorCPU>& integrator);

void integrator_rk4_step(std::unique_ptr<IntegratorCPU>& integrator);

void integrator_midpoint_step(std::unique_ptr<IntegratorCPU>& integrator);

void integrator_euler_cromer_step(std::unique_ptr<IntegratorCPU>& integrator);

void integrator_cd_step(std::unique_ptr<IntegratorCPU>& integrator);

extern "C" void cd_lu_chen_step(IntegratorCPU* integrator);
