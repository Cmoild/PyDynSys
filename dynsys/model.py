from . import _dynsys as core

class Model:

    def __init__(self):
        self.sim = core.SimulationCPU(0.01, [1., 1., 1.], 1.)
        self.sim.compileCode("""
        void lorenz(const float* X, const float* C, float* X_dot) {
                        X_dot[0] = 10. * (X[1] - X[0]);
                        X_dot[1] = X[0] * (28. - X[2]) - X[1];
                        X_dot[2] = X[0] * X[1] - 8. / 3. * X[2];
                    }
        """)
        self.sim.runSimulation(100, core.EULER)