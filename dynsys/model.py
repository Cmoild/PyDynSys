from . import _dynsys as core
import numpy


class Model:
    def __init__(self, dt: float, initial: list[float], params: numpy.ndarray):
        assert len(initial) == 3, "invalid array length"

        self.sim: core.SimulationCPU = core.SimulationCPU(dt, initial, params)

        self.sim.compileCode("""
        void lorenz(const float* X, const float* C, float* X_dot) {
                        X_dot[0] = C[0] * (X[1] - X[0]);
                        X_dot[1] = X[0] * (C[1] - X[2]) - X[1];
                        X_dot[2] = X[0] * X[1] - C[2] * X[2];
                    }
        """)

        self.num_constants: int = params.shape[0]

    def run(self, steps: int, integrator_type: str) -> numpy.ndarray:
        assert steps > 0, "steps must be positive int"
        assert integrator_type in ["euler", "rk4", "midpoint", "euler-cromer"], (
            "invalid integrator type"
        )

        select_integrator = {
            "euler": core.EULER,
            "rk4": core.RUNGE_KUTTA_4,
            "midpoint": core.RUNGE_KUTTA_4,
            "euler-cromer": core.EULER_CROMER,
        }

        return self.sim.runSimulation(steps, select_integrator[integrator_type])

    def bifurcation(
        self,
        steps: int,
        integrator_type: str,
        num_transition_points: int,
        parameter_idx: int,
        point_component_idx: int,
        min_max_dt: tuple[float, float, float],
    ):
        assert steps > 0, "steps must be positive int"
        assert integrator_type in ["euler", "rk4", "midpoint", "euler-cromer"], (
            "invalid integrator type"
        )

        select_integrator = {
            "euler": core.EULER,
            "rk4": core.RUNGE_KUTTA_4,
            "midpoint": core.RUNGE_KUTTA_4,
            "euler-cromer": core.EULER_CROMER,
        }
        return self.sim.createOneDimBifurcationDiagram(
            steps,
            select_integrator[integrator_type],
            parameter_idx,
            point_component_idx,
            self.num_constants,
            min_max_dt[0],
            min_max_dt[1],
            min_max_dt[2],
            num_transition_points,
        )

    def bifurcation2D(
        self,
        steps: int,
        integrator_type: str,
        num_transition_points: int,
        parameter_idx1: int,
        parameter_idx2: int,
        point_component_idx: int,
        min_max_dt1: tuple[float, float, float],
        min_max_dt2: tuple[float, float, float],
    ):
        assert steps > 0, "steps must be positive int"
        assert integrator_type in ["euler", "rk4", "midpoint", "euler-cromer"], (
            "invalid integrator type"
        )

        select_integrator = {
            "euler": core.EULER,
            "rk4": core.RUNGE_KUTTA_4,
            "midpoint": core.RUNGE_KUTTA_4,
            "euler-cromer": core.EULER_CROMER,
        }
        return self.sim.createTwoDimBifurcationDiagram(
            steps,
            select_integrator[integrator_type],
            parameter_idx1,
            parameter_idx2,
            point_component_idx,
            self.num_constants,
            min_max_dt1[0],
            min_max_dt1[1],
            min_max_dt1[2],
            min_max_dt2[0],
            min_max_dt2[1],
            min_max_dt2[2],
            num_transition_points,
        )
