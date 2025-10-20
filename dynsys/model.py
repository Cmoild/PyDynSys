from . import _dynsys as core
import typing
import numpy

class Model:

    def __init__(self, dt: float, initial: list[float], params: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]):
        assert len(initial) == 3, "invalid array length"

        self.sim = core.SimulationCPU(dt, initial, params)

        self.sim.compileCode("""
        void lorenz(const float* X, const float* C, float* X_dot) {
                        X_dot[0] = 10. * (X[1] - X[0]);
                        X_dot[1] = X[0] * (28. - X[2]) - X[1];
                        X_dot[2] = X[0] * X[1] - 8. / 3. * X[2];
                    }
        """)
    
    def run(self, steps: int, integrator_type: str):
        assert steps > 0, "steps must be positive int"
        assert integrator_type in ['euler', 'rk4', 'midpoint', 'euler-cromer'], "invalid integrator type"

        select_integrator = {
            'euler': core.EULER,
            'rk4': core.RUNGE_KUTTA_4,
            'midpoint': core.RUNGE_KUTTA_4,
            'euler_cromer': core.EULER_CROMER
        }

        return self.sim.runSimulation(steps, select_integrator[integrator_type])