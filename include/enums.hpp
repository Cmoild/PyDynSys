#pragma once

enum IntegratorType {
    EULER,
    RUNGE_KUTTA_4,
    MIDPOINT,
    EULER_CROMER,
    CD_METOD,
};

enum ComputeBackend { CPU, GPU };
