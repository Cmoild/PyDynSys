import numpy as np
from matplotlib import pyplot as plt
import dynsys
import time

dt = 0.01
initial = [1.0, 1.0, 1.0]

dsl = """
x' = sigma * (y - x)
y' = x * (rho - z) - y
z' = x * y - beta * z
"""

params = {"sigma": 10.0, "rho": 20.0, "beta": 8.0 / 3.0}

mod = dynsys.model.Model(dt=dt, initial=initial, params=params, code=dsl)

var_sys_dsl = """
dx' = -sigma * dx + sigma * dy
dy' = (rho - z) * dx - dy - x * dz
dz' = y * dx + x * dy - beta * dz
"""

bif2d = mod.lyapunov2d(
    steps=100000,
    num_transition_points=30000,
    integrator_type="rk4",
    variable_name="x",
    constants_dict={"sigma": (0.0, 100.0, 0.25), "rho": (0.0, 100.0, 0.25)},
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
