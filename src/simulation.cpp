#include <cassert>
#include <cstddef>
#include <memory>
#include <simulation.hpp>
#include <iostream>
#include <clustering/dbscan.h>
#include <unordered_set>
#include <algorithm>
#include <omp.h>

SimulationCPU::SimulationCPU(float deltaTime, std::array<float, 3> initConditions, float* C)
    : x_init(initConditions[0]), y_init(initConditions[1]), z_init(initConditions[2]),
      dt(deltaTime) {
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
    integrator->cur_step = 0;

    integrator->points = points;
    switch (iType) {
    case EULER: {
        for (size_t i = 0; i < num_points - 1; i++) {
            integrator_euler_step(integrator);
            // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
            //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
        }
    } break;
    case RUNGE_KUTTA_4: {
        for (size_t i = 0; i < num_points - 1; i++) {
            integrator_rk4_step(integrator);
            // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
            //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
        }
    } break;
    case MIDPOINT: {
        for (size_t i = 0; i < num_points - 1; i++) {
            integrator_midpoint_step(integrator);
            // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
            //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
        }
    } break;
    case EULER_CROMER: {
        for (size_t i = 0; i < num_points - 1; i++) {
            integrator_euler_cromer_step(integrator);
            // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
            //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
        }
    } break;
    case CD_METOD: {
        for (size_t i = 0; i < num_points - 1; i++) {
            // integrator_cd_step(integrator);
            cd_lu_chen_step(integrator.get());
            // std::cout << points[integrator->cur_step * 3] << ' ' << points[integrator->cur_step * 3 + 1] <<
            //     ' ' << points[integrator->cur_step * 3 + 2] << std::endl;
        }
    } break;
    }

    integrator->points = nullptr;
    return points;
}

std::shared_ptr<std::vector<float>> SimulationCPU::createOneDimBifurcationDiagram(
    const size_t num_points, const IntegratorType iType, const size_t parameterIdx,
    const size_t pointComponentIdx, const size_t numOfConstants, const float minValue,
    const float maxValue, const float deltaValue, const size_t numOfTransitionPoints) {
    size_t numThreads = (size_t)((maxValue - minValue) / deltaValue);
    auto out = std::make_shared<std::vector<float>>();
    std::vector<std::vector<float>> threadData(numThreads + 1);
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
        } break;
        case RUNGE_KUTTA_4: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_rk4_step(localIntegrator);
            }
        } break;
        case MIDPOINT: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_midpoint_step(localIntegrator);
            }
        } break;
        case EULER_CROMER: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_euler_cromer_step(localIntegrator);
            }
        } break;
        case CD_METOD: {
            for (size_t i = 0; i < num_points - 1; i++) {
                // integrator_cd_step(localIntegrator);
                cd_lu_chen_step(localIntegrator.get());
            }
        } break;
        }
        NDArray<float, 2> dbscanPoints({num_points - numOfTransitionPoints, 1});
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (size_t i = numOfTransitionPoints; i < num_points; i++) {
            dbscanPoints[i - numOfTransitionPoints][0] = points[i * 3 + pointComponentIdx];
        }
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        float eps = 0.3;
        size_t minPts = 15;
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        DBSCAN<float> dbscan(dbscanPoints, eps, minPts);
        dbscan.run();
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        const auto& labels = dbscan.labels();

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

        std::vector<float> localData;
        localData.reserve(centers.size() * 2);
        for (auto& c : centers) {
            localData.push_back(curConstant);
            localData.push_back(c);
        }
        // std::cout << (float)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        // std::cout << dbscan.nClusters() << ' ' << curConstant << std::endl;
        delete[] localIntegrator->C;
        delete[] points;

        threadData[thread] = std::move(localData);
    }
    for (auto& td : threadData) {
        out->insert(out->end(), td.begin(), td.end());
    }
    return out;
}

std::shared_ptr<std::vector<float>> SimulationCPU::createTwoDimBifurcatonDiagram(
    const size_t num_points, const IntegratorType iType, const size_t parameterIdx1,
    const size_t parameterIdx2, const size_t pointComponentIdx, const size_t numOfConstants,
    const float minValue1, const float maxValue1, const float deltaValue1, const float minValue2,
    const float maxValue2, const float deltaValue2, const size_t numOfTransitionPoints) {
    size_t numFirstValues = (size_t)((maxValue1 - minValue1) / deltaValue1) + 1;
    size_t numSecondValues = (size_t)((maxValue2 - minValue2) / deltaValue2) + 1;
    auto out = std::make_shared<std::vector<float>>(numFirstValues * numSecondValues);
    for (size_t threadOuterLoop = 0; threadOuterLoop < numFirstValues; threadOuterLoop++) {
        float curFirstConstant = minValue1 + (float)threadOuterLoop * deltaValue1;
        size_t numThreads = numSecondValues - 1;
#pragma omp parallel for
        for (size_t thread = 0; thread < numThreads + 1; thread++) {
            float curSecondConstant = minValue2 + (float)thread * deltaValue2;
            std::unique_ptr<IntegratorCPU> localIntegrator = std::make_unique<IntegratorCPU>();
            localIntegrator->C = new float[numOfConstants];
            std::memcpy(localIntegrator->C, integrator->C, sizeof(float) * numOfConstants);
            localIntegrator->cur_step = 0;
            localIntegrator->step = integrator->step;
            localIntegrator->func = integrator->func;
            localIntegrator->C[parameterIdx1] = curFirstConstant;
            localIntegrator->C[parameterIdx2] = curSecondConstant;
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
            } break;
            case RUNGE_KUTTA_4: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_rk4_step(localIntegrator);
                }
            } break;
            case MIDPOINT: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_midpoint_step(localIntegrator);
                }
            } break;
            case EULER_CROMER: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_euler_cromer_step(localIntegrator);
                }
            } break;
            case CD_METOD: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_cd_step(localIntegrator);
                }
            } break;
            }
            NDArray<float, 2> dbscanPoints({num_points - numOfTransitionPoints, 1});
            // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            for (size_t i = numOfTransitionPoints; i < num_points; i++) {
                dbscanPoints[i - numOfTransitionPoints][0] = points[i * 3 + pointComponentIdx];
            }
            // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            float eps = 0.3;
            size_t minPts = 15;
            // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            DBSCAN<float> dbscan(dbscanPoints, eps, minPts);
            dbscan.run();
            // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            // std::cout << (float)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
            // std::cout << dbscan.nClusters() << ' ' << curConstant << std::endl;
            (*out)[threadOuterLoop * numFirstValues + thread] =
                static_cast<float>(dbscan.nClusters());

            delete[] localIntegrator->C;
            delete[] points;
        }
        std::cout << "\r" << threadOuterLoop << "/" << numFirstValues;
        std::fflush(stdout);
    }
    std::cout << std::endl;
    return out;
}

std::shared_ptr<std::vector<float>> SimulationCPU::createOneDimLyapunovDiagram(
    const std::unique_ptr<SimulationCPU> jacobianFx,
    const std::unique_ptr<SimulationCPU> jacobianFy,
    const std::unique_ptr<SimulationCPU> jacobianFz, const size_t num_points,
    const IntegratorType iType, const size_t parameterIdx, const size_t pointComponentIdx,
    const size_t numOfConstants, const float minValue, const float maxValue, const float deltaValue,
    const size_t numOfTransitionPoints) {
    size_t numThreads = (size_t)((maxValue - minValue) / deltaValue);
    auto out = std::make_shared<std::vector<float>>();
    std::vector<std::vector<float>> threadData(numThreads + 1);
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
        } break;
        case RUNGE_KUTTA_4: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_rk4_step(localIntegrator);
            }
        } break;
        case MIDPOINT: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_midpoint_step(localIntegrator);
            }
        } break;
        case EULER_CROMER: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_euler_cromer_step(localIntegrator);
            }
        } break;
        case CD_METOD: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_cd_step(localIntegrator);
            }
        } break;
        }
        NDArray<float, 2> dbscanPoints({num_points - numOfTransitionPoints, 1});
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (size_t i = numOfTransitionPoints; i < num_points; i++) {
            dbscanPoints[i - numOfTransitionPoints][0] = points[i * 3 + pointComponentIdx];
        }
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        float eps = 0.3;
        size_t minPts = 15;
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        DBSCAN<float> dbscan(dbscanPoints, eps, minPts);
        dbscan.run();
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        const auto& labels = dbscan.labels();

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

        std::vector<float> localData;
        localData.reserve(centers.size() * 2);
        for (auto& c : centers) {
            localData.push_back(curConstant);
            localData.push_back(c);
        }
        // std::cout << (float)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        // std::cout << dbscan.nClusters() << ' ' << curConstant << std::endl;
        delete[] localIntegrator->C;
        delete[] points;

        threadData[thread] = std::move(localData);
    }
    for (auto& td : threadData) {
        out->insert(out->end(), td.begin(), td.end());
    }
    return out;
}
