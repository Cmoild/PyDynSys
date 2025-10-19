#include <backend/cpu/integrator_cpu.hpp>

void f_lorenz(float *params, float *out_params, float sigma,
                         float rho, float beta) {
    out_params[0] = sigma * (params[1] - params[0]);
    out_params[1] = params[0] * (rho - params[2]) - params[1];
    out_params[2] = params[0] * params[1] - beta * params[2];
}

void integrator_euler_step(std::unique_ptr<IntegratorCPU>& integrator) {
    size_t idx = 3 * integrator->cur_step;
    float cur_x = integrator->points[idx + 0];
    float cur_y = integrator->points[idx + 1];
    float cur_z = integrator->points[idx + 2];

    float derivs[3] = {0.0f};
    integrator->func(integrator->points + idx, integrator->C, derivs);

    size_t next_idx = 3 * (integrator->cur_step + 1);
    integrator->points[next_idx + 0] = cur_x + integrator->step * derivs[0];
    integrator->points[next_idx + 1] = cur_y + integrator->step * derivs[1];
    integrator->points[next_idx + 2] = cur_z + integrator->step * derivs[2];

    integrator->cur_step++;
}

void integrator_rk4_step(std::unique_ptr<IntegratorCPU>& integrator) {
    float k1[3], k2[3], k3[3], k4[3], tmp[3];

    size_t idx = 3 * integrator->cur_step;
    integrator->func(integrator->points + idx, integrator->C, k1);

    for (int i = 0; i < 3; i++)
        tmp[i] = integrator->points[idx + i] + 0.5f * integrator->step * k1[i];
    integrator->func(tmp, integrator->C, k2);

    for (int i = 0; i < 3; i++)
        tmp[i] = integrator->points[idx + i] + 0.5f * integrator->step * k2[i];
    integrator->func(tmp, integrator->C, k3);

    for (int i = 0; i < 3; i++)
        tmp[i] = integrator->points[idx + i] + integrator->step * k3[i];
    integrator->func(tmp, integrator->C, k4);

    size_t next_idx = 3 * (integrator->cur_step + 1);
    for (int i = 0; i < 3; i++) {
        integrator->points[next_idx + i] = integrator->points[idx + i] +
            (integrator->step / 6.0f) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }

    integrator->cur_step++;
}

void integrator_midpoint_step(std::unique_ptr<IntegratorCPU>& integrator) {
    float k1[3], k2[3], tmp[3];

    size_t idx = 3 * integrator->cur_step;

    integrator->func(integrator->points + idx, integrator->C, k1);

    for (int i = 0; i < 3; i++)
        tmp[i] = integrator->points[idx + i] + 0.5f * integrator->step * k1[i];

    integrator->func(tmp, integrator->C, k2);

    size_t next_idx = 3 * (integrator->cur_step + 1);
    for (int i = 0; i < 3; i++) {
        integrator->points[next_idx + i] = integrator->points[idx + i] +
                                          integrator->step * k2[i];
    }

    integrator->cur_step++;
}

void integrator_euler_cromer_step(std::unique_ptr<IntegratorCPU>& integrator) {
    size_t idx = 3 * integrator->cur_step;
    size_t next_idx = 3 * (integrator->cur_step + 1);

    float* current = integrator->points + idx;
    float* next = integrator->points + next_idx;

    float derivatives[3];

    integrator->func(current, integrator->C, derivatives);

    next[0] = current[0] + integrator->step * derivatives[0];

    float temp[3] = {next[0], current[1], current[2]};
    integrator->func(temp, integrator->C, derivatives);

    next[1] = current[1] + integrator->step * derivatives[1];

    temp[0] = next[0];
    temp[1] = next[1];
    temp[2] = current[2];
    integrator->func(temp, integrator->C, derivatives);

    next[2] = current[2] + integrator->step * derivatives[2];

    integrator->cur_step++;
}

void integrator_cd_step(std::unique_ptr<IntegratorCPU>& integrator) {
    size_t idx = 3 * integrator->cur_step;
    size_t next_idx = 3 * (integrator->cur_step + 1);

    float* current = integrator->points + idx;
    float* next = integrator->points + next_idx;

    float derivatives[3];

    integrator->func(current, integrator->C, derivatives);

    next[0] = current[0] + integrator->step / 2. * derivatives[0];

    float temp[3] = {next[0], current[1], current[2]};
    integrator->func(temp, integrator->C, derivatives);

    next[1] = current[1] + integrator->step / 2. * derivatives[1];

    temp[0] = next[0];
    temp[1] = next[1];
    temp[2] = current[2];
    integrator->func(temp, integrator->C, derivatives);

    next[2] = current[2] + integrator->step / 2. * derivatives[2];

    temp[2] = next[2];
    integrator->func(temp, integrator->C, derivatives);

    next[2] = next[2] + integrator->step / 2. * derivatives[2];

    temp[2] = next[2];
    integrator->func(temp, integrator->C, derivatives);

    next[1] = next[1] + integrator->step / 2. * derivatives[1];

    temp[1] = next[1];
    integrator->func(temp, integrator->C, derivatives);

    next[0] = next[0] + integrator->step / 2. * derivatives[0];

    integrator->cur_step++;
}
