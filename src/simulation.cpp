#include <cstddef>
#include <memory>
#include <simulation.hpp>
#include <iostream>
#include <clustering/dbscan.h>
#include <unordered_set>
#include <algorithm>

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

float* SimulationCPU::runSimulation(const size_t num_points, const IntegratorType iType) {
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

std::shared_ptr<std::vector<float>> SimulationCPU::createOneDimBifurcationDiagram(
    const size_t num_points, const IntegratorType iType, const size_t parameterIdx, const size_t pointComponentIdx,
    const size_t numOfConstants,
    const float minValue, const float maxValue, const float deltaValue, const size_t numOfTransitionPoints) {
    size_t numThreads = (size_t)((maxValue - minValue) / deltaValue);
    auto out = std::make_shared<std::vector<float>>();
    #pragma omp parallel for
    for (size_t thread = 0; thread < numThreads + 1; thread++) {
        float curConstant = minValue + (float)thread * deltaValue;
        std::unique_ptr<IntegratorCPU> localIntegrator = std::make_unique<IntegratorCPU>();
        localIntegrator->C = new float[numOfConstants];
        std::memcpy(localIntegrator->C, integrator->C, sizeof(float) * numOfConstants);
        localIntegrator->cur_step = 0;
        localIntegrator->step = integrator->step;
        localIntegrator->func = integrator->func;
        localIntegrator->C[parameterIdx] = curConstant;
        float* points = new float[3 * num_points];
        points[0] = x_init;
        points[1] = y_init;
        points[2] = z_init;

        localIntegrator->points = points;

        switch (iType) {
            case EULER: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_euler_step(localIntegrator);
                } 
            }
            break;
            case RUNGE_KUTTA_4: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_rk4_step(localIntegrator);
                } 
            }
            break;
            case MIDPOINT: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_midpoint_step(localIntegrator);
                } 
            }
            break;
            case EULER_CROMER: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_euler_cromer_step(localIntegrator);
                }
            }
            break;
            case CD_METOD: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_cd_step(localIntegrator);
                }
            }
            break;
        }
        NDArray<float, 2> dbscanPoints({num_points - numOfTransitionPoints, 1});
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (size_t i = numOfTransitionPoints; i < num_points; i++) {
            dbscanPoints[i - numOfTransitionPoints][0] = points[i * 3 + pointComponentIdx];
        }
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        float       eps        = 0.3;
        size_t      minPts     = 15;
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        DBSCAN<float> dbscan(dbscanPoints, eps, minPts);
        dbscan.run();
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        const auto &labels = dbscan.labels();

        std::unordered_set<int> labelsSet(labels.begin(), labels.end());
        std::vector<int> uniqueLabels(labelsSet.begin(), labelsSet.end());
        std::sort(uniqueLabels.begin(), uniqueLabels.end());

        std::vector<float> centers;

        for (int lab : uniqueLabels) {
            if (lab == dbscan.NOISY || lab == dbscan.UNCLASSIFIED)
                continue;

            std::vector<float> vals;
            for (size_t i = 0; i < labels.size(); ++i) {
                if (labels[i] == lab)
                    vals.push_back(dbscanPoints[i][0]);
            }

            float mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
            centers.push_back(mean);
        }
        for (auto& c : centers) {
            #pragma omp critical
            out->push_back(curConstant);
            #pragma omp critical
            out->push_back(c);
        }
        // std::cout << (float)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl; 
        // std::cout << dbscan.nClusters() << ' ' << curConstant << std::endl;
        delete [] localIntegrator->C;
        delete [] points;
    }
    return out;
}


