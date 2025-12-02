import numpy as np
from matplotlib import pyplot as plt
import dynsys
import time

dt = 0.01
initial = [0.1, 0.3, -0.6]

dsl = """
x' = a * (y - x)
y' = (1 - z) * x + c * y + u
z' = x * y - b * z
"""

params = {"a": 36.0, "b": 3.0, "c": 20.0, "u": 0}

mod = dynsys.model.Model(dt=dt, initial=initial, params=params, code=dsl)


var_sys_dsl = """
dx' = - a * dx + a * dy
dy' = (1. - z) * dx + c * dy - x * dz
dz' = y * dx + x * dy - b * dz
"""

bif2d = mod.lyapunov2d(
    steps=5000,
    integrator_type="rk4",
    variable_name="x",
    constants_dict={"a": (0.0, 100.0, 0.25), "b": (0.0, 100.0, 0.25)},
    var_sys_code=var_sys_dsl,
)

bif2d_2d = bif2d[:, :, 0]

plt.figure(figsize=(8, 6))
plt.imshow(
    bif2d_2d,
    extent=(0.0, 100.0, 0.0, 100.0),
    origin="lower",
    aspect="auto",
    cmap="plasma",
)
plt.colorbar(label="LE")
plt.xlabel("Parameter 1")
plt.ylabel("Parameter 2")
plt.title("2D lyapunov diagram")
plt.show()
