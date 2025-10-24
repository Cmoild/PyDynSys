from . import _dynsys as core
import numpy
from numpy.typing import NDArray
from typing import Literal
from . import dsl


class Model:
    def __init__(
        self, dt: float, initial: list[float], params: dict[str, float], code: str
    ):
        c_code, var_order, const_order = dsl.compile_dsl_to_c(code, params, "lorenz")
        self.var_order: list[str] = var_order
        self.const_order: list[str] = const_order

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
        integrator_type: Literal["euler", "rk4", "midpoint", "euler-cromer"],
    ) -> NDArray[numpy.float32]:
        assert steps > 0, "steps must be positive int"

        select_integrator = {
            "euler": core.EULER,
            "rk4": core.RUNGE_KUTTA_4,
            "midpoint": core.RUNGE_KUTTA_4,
            "euler-cromer": core.EULER_CROMER,
        }

        return self.sim.runSimulation(steps, select_integrator[integrator_type])

    def bifurcation1d(
        self,
        steps: int,
        integrator_type: Literal["euler", "rk4", "midpoint", "euler-cromer"],
        num_transition_points: int,
        constant_name: str,
        variable_name: str,
        min_max_dt: tuple[float, float, float],
    ) -> NDArray[numpy.float32]:
        assert steps > 0, "steps must be positive int"
        assert constant_name in self.const_order, "invalid name of constant"
        assert variable_name in self.var_order, "invalid name of variable"

        parameter_idx: int = self.const_order.index(constant_name)
        point_component_idx: int = self.var_order.index(variable_name)

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

    def bifurcation2d(
        self,
        steps: int,
        integrator_type: Literal["euler", "rk4", "midpoint", "euler-cromer"],
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

        parameter_idx1 = self.const_order.index(first_constant_name)
        parameter_idx2 = self.const_order.index(second_constant_name)

        point_component_idx = self.var_order.index(variable_name)

        min_max_dt1 = constants_dict[first_constant_name]
        min_max_dt2 = constants_dict[second_constant_name]

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
