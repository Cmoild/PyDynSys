#include <simulation.hpp>
#include <iostream>

SimulationCPU::SimulationCPU(float deltaTime, std::array<float, 3> initConditions, float* C) : 
    x_init(initConditions[0]), y_init(initConditions[1]), z_init(initConditions[2]),
    dt(deltaTime){
    integrator = std::make_unique<IntegratorCPU>();
    integrator->C = C;
    integrator->cur_step = 0;
    integrator->step = dt;
}

void SimulationCPU::compileCode(std::string userCode) {
    int err = try_jit(userCode, integrator->func, jit);
    if (err) {
        std::cerr << "JIT'ing error" << std::endl;
        return;
    }
}

float* SimulationCPU::runSimulation(size_t num_points, IntegratorType iType) {
    float* points = new float[3 * num_points];
    points[0] = x_init;
    points[1] = y_init;
    points[2] = z_init;

    integrator->points = points;
    switch (iType) {
        case EULER: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_euler_step(integrator);
                // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
                //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
            } 
        }
        break;
        case RUNGE_KUTTA_4: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_rk4_step(integrator);
                // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
                //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
            } 
        }
        break;
        case MIDPOINT: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_midpoint_step(integrator);
                // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
                //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
            } 
        }
        break;
        case EULER_CROMER: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_euler_cromer_step(integrator);
                // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
                //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
            }
        }
        break;
        case CD_METOD: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_cd_step(integrator);
                // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
                //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
            }
        }
        break;
    }

    return points;
}
