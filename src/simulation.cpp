#include "backend/cpu/integrator_cpu.hpp"
#include <cassert>
#include <cmath>
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

static inline int peakFinder(const float* points, const size_t num_points,
                             const size_t numOfTransitionPoints, const size_t pointComponentIdx,
                             std::unique_ptr<IntegratorCPU>& localIntegrator,
                             std::vector<float>& peakVals, std::vector<float>& intervalVals) {
    float latestPeakTime;
    bool firstPeakReady = false;
    // for (size_t i = 0; i < 15; i++) {
    //     std::cout << points[i] << ' ';
    //     if ((i + 1) % 3 == 0)
    //         std::cout << std::endl;
    // }
    for (size_t i = numOfTransitionPoints; i < num_points - 1; i++) {
        // std::cout << points[(i - 1) * 3 + pointComponentIdx] << ' '
        //           << points[i * 3 + pointComponentIdx] << ' '
        //           << points[(i + 1) * 3 + pointComponentIdx] << std::endl;
        if (std::isnan(points[i * 3 + pointComponentIdx]) ||
            std::isinf(points[i * 3 + pointComponentIdx]))
            return 1;
        if (points[i * 3 + pointComponentIdx] > points[(i - 1) * 3 + pointComponentIdx] &&
            points[i * 3 + pointComponentIdx] > points[(i + 1) * 3 + pointComponentIdx]) {
            // std::cout << "and what???" << std::endl;
            if (peakVals.size() == 0 && !firstPeakReady) {
                // firstPeakVal = points[i * 3 + pointComponentIdx];
                latestPeakTime = static_cast<float>(i) * localIntegrator->step;
                firstPeakReady = true;
            } else {
                peakVals.push_back(points[i * 3 + pointComponentIdx]);
                intervalVals.push_back(static_cast<float>(i) * localIntegrator->step -
                                       latestPeakTime);
                latestPeakTime = static_cast<float>(i) * localIntegrator->step;
            }
        }
    }

    if (peakVals.size() == 0)
        return 2;

    return 0;
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
        std::vector<float> peakVals(0, 0);
        std::vector<float> intervalVals(0, 0);
        peakFinder(points, num_points, numOfTransitionPoints, pointComponentIdx, localIntegrator,
                   peakVals, intervalVals);

        std::vector<float> localData(peakVals.size() * 3, 0.f);
        for (size_t i = 0; i < peakVals.size(); i++) {
            localData[i * 3] = curConstant;
            localData[i * 3 + 1] = peakVals[i];
            localData[i * 3 + 2] = intervalVals[i];
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
            std::vector<float> peakVals(0, 0);
            std::vector<float> intervalVals(0, 0);
            int peakFinderCode =
                peakFinder(points, num_points, numOfTransitionPoints, pointComponentIdx,
                           localIntegrator, peakVals, intervalVals);
            if (peakFinderCode == 0) {
                NDArray<float, 2> dbscanPoints({peakVals.size(), 2});
                for (size_t i = 0; i < peakVals.size(); i++) {
                    dbscanPoints[i][0] = peakVals[i];
                    dbscanPoints[i][1] = intervalVals[i];
                }
                float eps = 0.3;
                size_t minPts = 3;
                DBSCAN<float> dbscan(dbscanPoints, eps, minPts);
                dbscan.run();

                (*out)[thread * numFirstValues + threadOuterLoop] =
                    static_cast<float>(dbscan.nClusters());
            } else if (peakFinderCode == 1) {
                (*out)[thread * numFirstValues + threadOuterLoop] = std::nanf("");
            } else if (peakFinderCode == 2) {
                (*out)[thread * numFirstValues + threadOuterLoop] = 0;
            }

            delete[] localIntegrator->C;
            delete[] points;
        }
        std::cout << "\r" << threadOuterLoop + 1 << "/" << numFirstValues;
        std::fflush(stdout);
    }
    std::cout << std::endl;
    return out;
}

static inline float computeLamda(std::unique_ptr<IntegratorCPU>& integrator,
                                 std::vector<float>& varSysConsts, float* varSysVector,
                                 integratedFunc varSysFunc) {
    float varSysStep = integrator->step;
    // dt * Jacobian @ varSysVect
    // float tmp[3] = {
    //     varSysStep * (-a * varSysVector[0] + a * varSysVector[1]),
    //     varSysStep *
    //         ((1.f - cur_z) * varSysVector[0] + c * varSysVector[1] - cur_x * varSysVector[2]),
    //     varSysStep * (cur_y * varSysVector[0] + cur_x * varSysVector[1] - b * varSysVector[2]),
    // };
    float tmp[3];
    varSysFunc(varSysVector, varSysConsts.data(), tmp);

    varSysVector[0] += tmp[0] * varSysStep;
    varSysVector[1] += tmp[1] * varSysStep;
    varSysVector[2] += tmp[2] * varSysStep;

    // normalize(varSysVect)
    float norm = std::sqrtf(varSysVector[0] * varSysVector[0] + varSysVector[1] * varSysVector[1] +
                            varSysVector[2] * varSysVector[2]);
    varSysVector[0] /= norm;
    varSysVector[1] /= norm;
    varSysVector[2] /= norm;

    // if (std::isnan(std::logf(std::max<float>(norm, 1.)))) {
    //     std::cout << norm << std::endl;
    //     std::cout << varSysVector[0] << ' ' << varSysVector[1] << ' ' << varSysVector[2];
    // }

    return std::logf(norm);
}

std::shared_ptr<std::vector<float>> SimulationCPU::createOneDimLyapunovDiagram(
    const size_t num_points, const IntegratorType iType, const size_t parameterIdx,
    const size_t numOfConstants, const float minValue, const float maxValue, const float deltaValue,
    const std::array<size_t, 3> xyzOrder, std::string varSysCode) {
    size_t numThreads = (size_t)((maxValue - minValue) / deltaValue);
    auto out = std::make_shared<std::vector<float>>((numThreads + 1) * 2);
    std::default_random_engine generator(42);
    std::normal_distribution<float> distribution(0.f, 1.f);
    std::unique_ptr<llvm::orc::LLJIT> varSysJIT;
    integratedFunc varSysFunc = nullptr;
    int err = try_jit(varSysCode, varSysFunc, varSysJIT, "varSysFunc");
    if (err) {
        std::cerr << "JIT'ing error" << std::endl;
        return out;
    }
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
        std::vector<float> varSysConsts(numOfConstants + 3);
        for (size_t i = 0; i < numOfConstants; i++) {
            varSysConsts[i] = localIntegrator->C[i];
        }

        switch (iType) {
        case EULER: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_euler_step(localIntegrator);
                size_t idx = 3 * localIntegrator->cur_step;
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                lambdaSum += computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
            }
        } break;
        case RUNGE_KUTTA_4: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_rk4_step(localIntegrator);
                size_t idx = 3 * localIntegrator->cur_step;
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                lambdaSum += computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
            }
        } break;
        case MIDPOINT: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_midpoint_step(localIntegrator);
                size_t idx = 3 * localIntegrator->cur_step;
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                lambdaSum += computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
            }
        } break;
        case EULER_CROMER: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_euler_cromer_step(localIntegrator);
                size_t idx = 3 * localIntegrator->cur_step;
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                lambdaSum += computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
            }
        } break;
        case CD_METOD: {
            for (size_t i = 0; i < num_points - 1; i++) {
                integrator_cd_step(localIntegrator);
                size_t idx = 3 * localIntegrator->cur_step;
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                lambdaSum += computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
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

std::shared_ptr<std::vector<float>> SimulationCPU::createTwoDimLyapunovDiagram(
    const size_t num_points, const IntegratorType iType, const size_t parameterIdx1,
    const size_t parameterIdx2, const size_t numOfConstants, const float minValue1,
    const float maxValue1, const float deltaValue1, const float minValue2, const float maxValue2,
    const float deltaValue2, const std::array<size_t, 3> xyzOrder, std::string varSysCode) {
    size_t numFirstValues = (size_t)((maxValue1 - minValue1) / deltaValue1) + 1;
    size_t numSecondValues = (size_t)((maxValue2 - minValue2) / deltaValue2) + 1;
    auto out = std::make_shared<std::vector<float>>(numFirstValues * numSecondValues);
    std::default_random_engine generator(42);
    std::normal_distribution<float> distribution(0.f, 1.f);
    std::unique_ptr<llvm::orc::LLJIT> varSysJIT;
    integratedFunc varSysFunc = nullptr;
    int err = try_jit(varSysCode, varSysFunc, varSysJIT, "varSysFunc");
    if (err) {
        std::cerr << "JIT'ing error" << std::endl;
        return out;
    }

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
            float varSysVector[3] = {
                distribution(generator),
                distribution(generator),
                distribution(generator),
            };
            float lambdaSum = 0.f;
            std::vector<float> varSysConsts(numOfConstants + 3);
            for (size_t i = 0; i < numOfConstants; i++) {
                varSysConsts[i] = localIntegrator->C[i];
            }

            switch (iType) {
            case EULER: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_euler_step(localIntegrator);
                    size_t idx = 3 * localIntegrator->cur_step;
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                    lambdaSum +=
                        computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
                }
            } break;
            case RUNGE_KUTTA_4: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_rk4_step(localIntegrator);
                    size_t idx = 3 * localIntegrator->cur_step;
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                    lambdaSum +=
                        computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
                }
            } break;
            case MIDPOINT: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_midpoint_step(localIntegrator);
                    size_t idx = 3 * localIntegrator->cur_step;
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                    lambdaSum +=
                        computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
                }
            } break;
            case EULER_CROMER: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_euler_cromer_step(localIntegrator);
                    size_t idx = 3 * localIntegrator->cur_step;
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                    lambdaSum +=
                        computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
                }
            } break;
            case CD_METOD: {
                for (size_t i = 0; i < num_points - 1; i++) {
                    integrator_cd_step(localIntegrator);
                    size_t idx = 3 * localIntegrator->cur_step;
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[0]] = points[idx + xyzOrder[0]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[1]] = points[idx + xyzOrder[1]];
                    varSysConsts[varSysConsts.size() - 3 + xyzOrder[2]] = points[idx + xyzOrder[2]];
                    lambdaSum +=
                        computeLamda(localIntegrator, varSysConsts, varSysVector, varSysFunc);
                }
            } break;
            }

            (*out)[thread * numFirstValues + threadOuterLoop] =
                lambdaSum / ((float)localIntegrator->cur_step * localIntegrator->step);

            delete[] localIntegrator->C;
            delete[] points;
        }
        std::cout << "\r" << threadOuterLoop + 1 << "/" << numFirstValues;
        std::fflush(stdout);
    }
    std::cout << std::endl;
    return out;
}
