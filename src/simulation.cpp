#include <cassert>
#include <cstddef>
#include <memory>
#include <simulation.hpp>
#include <iostream>
#include <clustering/dbscan.h>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <omp.h>

SimulationCPU::SimulationCPU(float deltaTime, std::array<float, 3> initConditions, float* C,
                             size_t numConsts)
    : x_init(initConditions[0]), y_init(initConditions[1]), z_init(initConditions[2]),
      dt(deltaTime) {
    this->constants = std::vector<float>(C, C + numConsts);
    integrator = std::make_unique<IntegratorCPU>();
    integrator->C = this->constants.data();
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
        }
    } break;
    case RUNGE_KUTTA_4: {
        for (size_t i = 0; i < num_points - 1; i++) {
            integrator_rk4_step(integrator);
        }
    } break;
    case MIDPOINT: {
        for (size_t i = 0; i < num_points - 1; i++) {
            integrator_midpoint_step(integrator);
        }
    } break;
    case EULER_CROMER: {
        for (size_t i = 0; i < num_points - 1; i++) {
            integrator_euler_cromer_step(integrator);
        }
    } break;
    case CD_METOD: {
        for (size_t i = 0; i < num_points - 1; i++) {
            cd_lu_chen_step(integrator.get());
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
                cd_lu_chen_step(localIntegrator.get());
            }
        } break;
        }
        NDArray<float, 2> dbscanPoints({num_points - numOfTransitionPoints, 1});
        for (size_t i = numOfTransitionPoints; i < num_points; i++) {
            dbscanPoints[i - numOfTransitionPoints][0] = points[i * 3 + pointComponentIdx];
        }
        float eps = 0.3;
        size_t minPts = 15;
        DBSCAN<float> dbscan(dbscanPoints, eps, minPts);
        dbscan.run();
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
            for (size_t i = numOfTransitionPoints; i < num_points; i++) {
                dbscanPoints[i - numOfTransitionPoints][0] = points[i * 3 + pointComponentIdx];
            }
            float eps = 0.3;
            size_t minPts = 15;
            DBSCAN<float> dbscan(dbscanPoints, eps, minPts);
            dbscan.run();

            (*out)[thread * numFirstValues + threadOuterLoop] =
                static_cast<float>(dbscan.nClusters());

            delete[] localIntegrator->C;
            delete[] points;
        }
        std::cout << "\r" << threadOuterLoop + 1 << "/" << numFirstValues;
        std::fflush(stdout);
    }
    std::cout << std::endl;
    return out;
}

static inline float computeLambdaForLuChen(std::unique_ptr<IntegratorCPU>& integrator,
                                           const std::array<size_t, 3> xyzOrder,
                                           const std::array<size_t, 4> constOrder,
                                           float* varSysVector) {
    float a, b, c, u;
    a = integrator->C[constOrder[0]];
    b = integrator->C[constOrder[1]];
    c = integrator->C[constOrder[2]];
    u = integrator->C[constOrder[3]];

    size_t idx = 3 * integrator->cur_step;
    float cur_x = integrator->points[idx + xyzOrder[0]];
    float cur_y = integrator->points[idx + xyzOrder[1]];
    float cur_z = integrator->points[idx + xyzOrder[2]];

    // dt * Jacobian @ varSysVect
    float tmp[3] = {
        integrator->step * (-a * varSysVector[0] + a * varSysVector[1]),
        integrator->step *
            ((1.f - cur_z) * varSysVector[0] + c * varSysVector[1] - cur_x * varSysVector[2]),
        integrator->step *
            (cur_y * varSysVector[0] + cur_x * varSysVector[1] - b * varSysVector[2]),
    };

    varSysVector[0] += tmp[0];
    varSysVector[1] += tmp[1];
    varSysVector[2] += tmp[2];

    // normalize(varSysVect)
    float norm = std::sqrtf(varSysVector[0] * varSysVector[0] + varSysVector[1] * varSysVector[1] +
                            varSysVector[2] * varSysVector[2]);
    varSysVector[0] /= norm;
    varSysVector[1] /= norm;
    varSysVector[2] /= norm;

    return std::logf(norm);
}

std::shared_ptr<std::vector<float>> SimulationCPU::createOneDimLyapunovDiagram(
    const size_t num_points, const IntegratorType iType, const size_t parameterIdx,
    const size_t numOfConstants, const float minValue, const float maxValue, const float deltaValue,
    const std::array<size_t, 3> xyzOrder, const std::array<size_t, 4> constOrder) {
    size_t numThreads = (size_t)((maxValue - minValue) / deltaValue);
    auto out = std::make_shared<std::vector<float>>((numThreads + 1) * 2);
    std::default_random_engine generator(42);
    std::normal_distribution<float> distribution(0.f, 1.f);
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
        float varSysVector[3] = {
            distribution(generator),
            distribution(generator),
            distribution(generator),
        };
        float lambdaSum = 0.f;

        switch (iType) {
        case EULER: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_euler_step(localIntegrator);
                lambdaSum +=
                    computeLambdaForLuChen(localIntegrator, xyzOrder, constOrder, varSysVector);
            }
        } break;
        case RUNGE_KUTTA_4: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_rk4_step(localIntegrator);
                lambdaSum +=
                    computeLambdaForLuChen(localIntegrator, xyzOrder, constOrder, varSysVector);
            }
        } break;
        case MIDPOINT: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_midpoint_step(localIntegrator);
                lambdaSum +=
                    computeLambdaForLuChen(localIntegrator, xyzOrder, constOrder, varSysVector);
            }
        } break;
        case EULER_CROMER: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_euler_cromer_step(localIntegrator);
                lambdaSum +=
                    computeLambdaForLuChen(localIntegrator, xyzOrder, constOrder, varSysVector);
            }
        } break;
        case CD_METOD: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_cd_step(localIntegrator);
                lambdaSum +=
                    computeLambdaForLuChen(localIntegrator, xyzOrder, constOrder, varSysVector);
            }
        } break;
        }
        delete[] localIntegrator->C;
        delete[] points;

#pragma omp critical
        {
            (*out)[2 * thread] = curConstant;
            (*out)[2 * thread + 1] =
                lambdaSum / ((float)localIntegrator->cur_step * localIntegrator->step);
        }
    }
    return out;
}
