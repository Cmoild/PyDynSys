import matplotlib
from matplotlib import pyplot as plt
from . import _dynsys as core
import numpy
from numpy.typing import NDArray
from typing import Literal
import warnings
from . import dsl


class Model:
    def __init__(
        self, dt: float, initial: list[float], params: dict[str, float], code: str
    ):
        c_code, var_order, const_order = dsl.compile_dsl_to_c(code, params, "lorenz")
        self.var_order: list[str] = var_order
        self.const_order: list[str] = const_order
        self.code: str = code

        params_float_list: list[float] = []
        for c in const_order:
            params_float_list.append(params[c])

        self.sim: core.SimulationCPU = core.SimulationCPU(
            dt, initial, numpy.array(params_float_list, dtype=numpy.float32)
        )

        declared_math_functions = """
            float cosf(float);
            float sinf(float);
            float expf(float);
            float logf(float);
        """
        self.sim.compileCode(declared_math_functions + c_code)

        self.num_constants: int = len(params.keys())

    def run(
        self,
        steps: int,
        integrator_type: Literal["euler", "rk4", "midpoint", "euler-cromer", "cd"],
        show_plot: bool = False,
    ) -> NDArray[numpy.float32]:
        assert steps > 0, "steps must be positive int"

        select_integrator = {
            "euler": core.EULER,
            "rk4": core.RUNGE_KUTTA_4,
            "midpoint": core.RUNGE_KUTTA_4,
            "euler-cromer": core.EULER_CROMER,
            "cd": core.CD,
        }

        traj = self.sim.runSimulation(steps, select_integrator[integrator_type])

        if not show_plot:
            return traj

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        if integrator_type != "cd":
            ax.plot(
                traj[:, self.var_order.index("x")],
                traj[:, self.var_order.index("y")],
                traj[:, self.var_order.index("z")],
                c="r",
                linewidth=0.5,
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                c="r",
                linewidth=0.5,
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Attractor, method: {integrator_type}")

        plt.show()
        return traj

    def bifurcation1d(
        self,
        steps: int,
        integrator_type: Literal["euler", "rk4", "midpoint", "euler-cromer", "cd"],
        num_transition_points: int,
        constant_name: str,
        variable_name: str,
        min_max_dt: tuple[float, float, float],
    ) -> NDArray[numpy.float32]:
        assert steps > 0, "steps must be positive int"
        assert constant_name in self.const_order, "invalid name of constant"
        assert variable_name in self.var_order, "invalid name of variable"

        parameter_idx: int = (
            self.const_order.index(constant_name)
            if integrator_type != "cd"
            else ["a", "b", "c", "u"].index(constant_name)
        )
        point_component_idx: int = (
            self.var_order.index(variable_name)
            if integrator_type != "cd"
            else ["x", "y", "z"].index(variable_name)
        )

        select_integrator = {
            "euler": core.EULER,
            "rk4": core.RUNGE_KUTTA_4,
            "midpoint": core.RUNGE_KUTTA_4,
            "euler-cromer": core.EULER_CROMER,
            "cd": core.CD,
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

    def bifurcation2d(
        self,
        steps: int,
        integrator_type: Literal["euler", "rk4", "midpoint", "euler-cromer", "cd"],
        num_transition_points: int,
        variable_name: str,
        constants_dict: dict[str, tuple[float, float, float]],
    ) -> NDArray[numpy.float32]:
        assert steps > 0, "steps must be positive int"
        assert len(constants_dict.keys()) == 2, "too many constants"

        first_constant_name, second_constant_name = constants_dict.keys()

        assert first_constant_name in self.const_order, (
            "invalid name of the first constant"
        )
        assert second_constant_name in self.const_order, (
            "invalid name of the second constant"
        )
        assert variable_name in self.var_order, "invalid name of variable"

        parameter_idx1 = (
            self.const_order.index(first_constant_name)
            if integrator_type != "cd"
            else ["a", "b", "c", "u"].index(first_constant_name)
        )
        parameter_idx2 = (
            self.const_order.index(second_constant_name)
            if integrator_type != "cd"
            else ["a", "b", "c", "u"].index(second_constant_name)
        )

        point_component_idx = (
            self.var_order.index(variable_name)
            if integrator_type != "cd"
            else ["x", "y", "z"].index(variable_name)
        )

        min_max_dt1 = constants_dict[first_constant_name]
        min_max_dt2 = constants_dict[second_constant_name]

        select_integrator = {
            "euler": core.EULER,
            "rk4": core.RUNGE_KUTTA_4,
            "midpoint": core.RUNGE_KUTTA_4,
            "euler-cromer": core.EULER_CROMER,
            "cd": core.CD,
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

    def lyapunov1d(
        self,
        steps: int,
        integrator_type: Literal["euler", "rk4", "midpoint", "euler-cromer", "cd"],
        constant_name: str,
        min_max_dt: tuple[float, float, float],
    ) -> NDArray[numpy.float32]:
        warnings.warn("Works only with Lu Chen system", UserWarning, 2)
        if (
            self.code
            != """
x' = a * (y - x)
y' = (1 - z) * x + c * y + u
z' = x * y - b * z
"""
        ):
            raise NotImplementedError

        assert steps > 0, "steps must be positive int"
        assert constant_name in self.const_order, "invalid name of constant"

        parameter_idx: int = (
            self.const_order.index(constant_name)
            if integrator_type != "cd"
            else ["a", "b", "c", "u"].index(constant_name)
        )

        select_integrator = {
            "euler": core.EULER,
            "rk4": core.RUNGE_KUTTA_4,
            "midpoint": core.RUNGE_KUTTA_4,
            "euler-cromer": core.EULER_CROMER,
            "cd": core.CD,
        }

        xyz_order = (
            [
                self.var_order.index("x"),
                self.var_order.index("y"),
                self.var_order.index("z"),
            ]
            if integrator_type != "cd"
            else [0, 1, 2]
        )
        const_order = (
            [
                self.const_order.index("a"),
                self.const_order.index("b"),
                self.const_order.index("c"),
                self.const_order.index("u"),
            ]
            if integrator_type != "cd"
            else [0, 1, 2, 3]
        )

        return self.sim.createOneDimLyapunovDiagram(
            steps,
            select_integrator[integrator_type],
            parameter_idx,
            self.num_constants,
            min_max_dt[0],
            min_max_dt[1],
            min_max_dt[2],
            xyz_order,
            const_order,
        )
